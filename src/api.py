# -*- coding: utf-8 -*-
"""
src/api.py —— REST API 服务
============================
供 MES/ERP 等外部系统集成调用诊断能力：

  GET  /api/health                健康检查
  GET  /api/v1/meta/features      特征字段元信息（顺序、含义）
  POST /api/v1/diagnose           单条诊断（带钢原始记录：6参数+3曲线）
  POST /api/v1/diagnose/batch     批量诊断（JSON 数组）
  POST /api/v1/diagnose/batch/file 批量诊断（CSV 文件上传）
  POST /api/v1/query              LLM 知识问答/Agent 诊断（需 DEEPSEEK_API_KEY）

启动：python -m uvicorn src.api:app --host 0.0.0.0 --port 8000
"""

import io
import json
import sys
import threading
import uuid
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from pydantic import BaseModel, Field

# 兼容 `python -m uvicorn src.api:app` 与直接运行两种方式
sys.path.insert(0, str(Path(__file__).resolve().parent))

from logger import get_logger  # noqa: E402
from service import (
    DEEPSEEK_API_KEY,
    MEMORY,
    MILVUS_DB,
    batch_predict,
    init_resources,
    predict_single,
)

app = FastAPI(
    title="带钢宽度缺陷智能诊断系统 API",
    description="提供故障分类、批量诊断与知识问答能力，供 MES/ERP 等外部系统集成。",
    version="1.0.0",
)

logger = get_logger("api")

_lock = threading.Lock()
_resources = None


def get_resources():
    """惰性加载模型/检索器/Agent（首次请求时加载，之后复用）。"""
    global _resources
    if _resources is None:
        with _lock:
            if _resources is None:
                logger.info("API 首次请求，开始初始化资源...")
                _resources = init_resources()
    return _resources


class DiagnoseRequest(BaseModel):
    strip: Dict[str, Any] = Field(
        ...,
        description="单条带钢记录：STRIPNO(可选) + 6个工艺参数(FMWTARGETHOT/RDWTARGETTOTAL/"
                   "PDIWIDTHTOL/FMWIDTHACTHOT/RMWIDTHACTHOT/FMWTARGETCOL) + 3段全长宽度曲线"
                   "(RMDATASEQ/FMDATASEQ/DCDATASEQ，逗号分隔数值字符串)",
    )


class BatchDiagnoseRequest(BaseModel):
    strips: List[Dict[str, Any]] = Field(
        ..., min_length=1,
        description="多条带钢记录组成的数组（字段同 DiagnoseRequest.strip）；"
                   "大批量建议改用 /diagnose/batch/file 上传 CSV 文件",
    )


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    features: Optional[str] = Field("", description="单条带钢记录 JSON 字符串（可选，字段同 /diagnose 的 strip）")
    session_id: Optional[str] = Field(
        None,
        description="会话 ID（32 位十六进制）。提供后由 API 自动读写 MEMORY（Redis/文件）"
        "持久化对话历史与批量诊断状态，外部系统无需每次传入完整历史。",
    )
    conversation_history: Optional[List[Dict[str, str]]] = Field(
        [], description="多轮对话历史（可选），格式 [{\"role\": \"user\", \"content\": \"...\"}]"
    )
    batch_done: Optional[bool] = Field(
        None, description="是否已执行批量诊断（可选）；未提供时优先从 MEMORY 读取"
    )
    batch_summary: Optional[str] = Field(
        None, description="批量诊断结果摘要 JSON（可选）；未提供时优先从 MEMORY 读取"
    )
    csv_path: Optional[str] = Field(
        None, description="已上传 CSV 的服务器路径（可选）。提供后 Agent 可对整批数据做批量诊断并支持追问"
    )


def _prob_map(model_info, probs) -> Dict[str, float]:
    if not probs:
        return {}
    label_mapping = model_info['label_mapping']
    return {
        label_mapping.get(i, str(i)): round(float(p), 6)
        for i, p in enumerate(probs)
    }


@app.get("/api/health")
def health():
    get_resources()  # 主动触发懒加载，避免前端一直显示"初始化中..."
    return {
        "status": "ok",
        "service": "steel-width-defect-diagnosis",
        "resources_loaded": _resources is not None,
        "redis_mode": MEMORY.mode,
        "llm_configured": bool(DEEPSEEK_API_KEY),
    }


@app.get("/api/v1/meta/features")
def meta_features():
    model_info, _retriever, _feature_names, _agent = get_resources()
    return {
        "n_features": len(model_info['feature_names']),
        "feature_names": model_info['feature_names'],
        "fault_types": list(model_info['label_mapping'].values()),
    }


@app.get("/api/v1/diag/history")
def diag_history(
    fault_cn: Optional[str] = None,
    days: Optional[int] = None,
    limit: int = 50,
):
    """按故障类型/时间范围查询单卷诊断历史（需启用 MySQL；未启用返回空列表）。"""
    recs = MEMORY.get_diag_history(
        fault_cn=fault_cn or None,
        days=days,
        limit=min(int(limit), 200),
    )
    return {"total": len(recs), "records": recs}


@app.post("/api/v1/diagnose")
def diagnose(req: DiagnoseRequest):
    model_info, _r, _f, _a = get_resources()
    try:
        sample_df = pd.DataFrame([req.strip])
        missing = [c for c in model_info['feature_names'] if c not in sample_df.columns]
        if missing:
            raise HTTPException(status_code=400, detail=f"缺少必要字段: {missing}")
        diag = predict_single(model_info, sample_df)
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        logger.error("单条诊断失败: %s", e)
        raise HTTPException(status_code=500, detail=f"诊断失败: {e}")
    probs = diag.get("probs") or []
    imp = model_info.get('importance_cache', {}).get(diag['fault_cn'], [])[:10]
    return {
        "request_id": uuid.uuid4().hex[:12],
        "fault_cn": diag['fault_cn'],
        "fault_desc": diag['fault_desc'],
        "confidence": diag.get('confidence'),
        "probabilities": _prob_map(model_info, probs),
        "needs_review": diag.get('needs_review', False),
        "evidence": diag.get('evidence', ''),
        "top_features": [
            {"feature": f, "importance": round(float(w), 6)} for f, w in imp
        ],
    }


def _run_batch_df(df: pd.DataFrame):
    model_info, _r, _f, _a = get_resources()
    results = batch_predict(model_info, df)
    distribution: Dict[str, int] = {}
    needs_review = 0
    out = []
    for i, r in enumerate(results):
        fault_cn = r['fault_cn']
        distribution[fault_cn] = distribution.get(fault_cn, 0) + 1
        if r.get('needs_review'):
            needs_review += 1
        probs = r.get("probs") or []
        out.append({
            "index": i,
            "stripno": r.get('stripno', ''),
            "fault_cn": fault_cn,
            "fault_desc": r['fault_desc'],
            "confidence": r.get('confidence'),
            "needs_review": bool(r.get('needs_review')),
            "evidence": r.get('evidence', ''),
            "features_named": {
                name: val for name, val in zip(
                    ["FMWTARGETHOT", "RDWTARGETTOTAL", "PDIWIDTHTOL",
                     "FMWIDTHACTHOT", "RMWIDTHACTHOT", "FMWTARGETCOL",
                     "is_FM", "is_RM", "is_DM", "is_HT", "is_WS", "is_NA"],
                    r.get('feat_vector12', []))
            },
        })
    return {
        "request_id": uuid.uuid4().hex[:12],
        "total": len(out),
        "distribution": distribution,
        "needs_review_count": needs_review,
        "results": out,
    }


@app.post("/api/v1/diagnose/batch")
def diagnose_batch(req: BatchDiagnoseRequest):
    try:
        return _run_batch_df(pd.DataFrame(req.strips))
    except Exception as e:  # noqa: BLE001
        logger.error("批量诊断失败: %s", e)
        raise HTTPException(status_code=500, detail=f"批量诊断失败: {e}")


@app.post("/api/v1/diagnose/batch/file")
async def diagnose_batch_file(file: UploadFile = File(...)):
    raw = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception:
        try:
            df = pd.read_csv(io.BytesIO(raw), encoding="gbk")
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=f"CSV 解析失败: {e}")
    model_info, _r, _f, _a = get_resources()
    missing = [c for c in model_info['feature_names'] if c not in df.columns]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"CSV 缺少特征列: {missing}；需要列: {model_info['feature_names']}",
        )
    return _run_batch_df(df)


# ==================== CSV 上传（供 Agent 会话式批量诊断） ====================
_CSV_UPLOAD_DIR = Path(__file__).resolve().parent.parent / "data" / "tmp_uploads"


@app.post("/api/v1/csv/upload")
async def csv_upload(file: UploadFile = File(...)):
    """保存上传的 CSV 到 data/tmp_uploads/，返回服务器路径供 /query 的 csv_path 使用。
    校验是否包含 12 个特征列，供前端即时反馈。"""
    _CSV_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    raw = await file.read()
    saved_name = f"{uuid.uuid4().hex}.csv"
    saved_path = _CSV_UPLOAD_DIR / saved_name
    saved_path.write_bytes(raw)

    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception:
        try:
            df = pd.read_csv(io.BytesIO(raw), encoding="gbk")
        except Exception as e:  # noqa: BLE001
            saved_path.unlink(missing_ok=True)
            raise HTTPException(status_code=400, detail=f"CSV 解析失败: {e}")

    model_info, _r, _f, _a = get_resources()
    feature_names = model_info['feature_names']
    missing = [c for c in feature_names if c not in df.columns]

    return {
        "csv_path": str(saved_path),
        "file_name": file.filename or saved_name,
        "rows": int(len(df)),
        "columns": list(df.columns),
        "missing_features": missing,
        "feature_names": feature_names,
    }


@app.post("/api/v1/query")
def query(req: QueryRequest):
    if not DEEPSEEK_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="未配置 DEEPSEEK_API_KEY，无法使用 LLM 知识问答（诊断接口不受影响）",
        )
    sid = (req.session_id or "").strip()
    if sid and not MEMORY.is_valid_sid(sid):
        raise HTTPException(
            status_code=400,
            detail="session_id 必须为 32 位十六进制字符串（可由客户端生成或省略以使用无状态模式）",
        )

    # ---------- 会话与批量状态：优先 MEMORY，其次请求体 ----------
    history = list(req.conversation_history or [])
    if sid and not history:
        history = MEMORY.get_messages(sid)
    batch_state = MEMORY.get_batch(sid) if sid else {}
    batch_done = req.batch_done if req.batch_done is not None else batch_state.get("batch_done", False)
    batch_summary = req.batch_summary if req.batch_summary else batch_state.get("batch_summary", "")
    # csv_path：请求体优先，其次 MEMORY（支持跨轮次沿用已上传文件）
    csv_path = req.csv_path or batch_state.get("csv_path", "")

    _m, _r, _f, agent_graph = get_resources()
    initial_state = {
        "query": req.question,
        "features": req.features or "",
        "csv_path": csv_path or "",
        "conversation_history": history,
        "intermediate_steps": [],
        "final_answer": "",
        "step_count": 0,
        "batch_done": bool(batch_done),
        "batch_summary": batch_summary or "",
    }
    try:
        final_state = agent_graph.invoke(initial_state)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"问答失败: {e}")

    answer = final_state.get("final_answer", "未能生成回答。")

    # ---------- 持久化：有 session_id 时自动写回 MEMORY ----------
    if sid:
        history = history + [
            {"role": "user", "content": req.question},
            {"role": "assistant", "content": answer},
        ]
        MEMORY.save_messages(sid, history)
        MEMORY.save_batch(sid, {
            "batch_done": final_state.get("batch_done", False),
            "batch_summary": final_state.get("batch_summary", ""),
            "csv_path": csv_path or "",
        })

    steps = [
        {"tool": tool, "observation": obs}
        for tool, obs in final_state.get("intermediate_steps", [])
    ]
    return {
        "request_id": uuid.uuid4().hex[:12],
        "session_id": sid or None,
        "answer": answer,
        "steps": steps,
        "batch_done": final_state.get("batch_done", False),
        "batch_summary": final_state.get("batch_summary", ""),
        "csv_path": csv_path or "",
        "history_length": len(history) if sid else len(req.conversation_history or []),
    }


# ============================================================
# 知识库浏览（只读，无需加载模型/LLM）
# ============================================================
_mkv_client = None
_mkv_lock = threading.Lock()


def _get_mkclient():
    """惰性获取 Milvus 客户端（知识库路由独立加载，不阻塞诊断资源）。"""
    global _mkv_client
    if _mkv_client is None:
        with _mkv_lock:
            if _mkv_client is None:
                try:
                    from pymilvus import MilvusClient
                    _mkv_client = MilvusClient(MILVUS_DB)
                    _mkv_client.load_collection("rag_knowledge")
                except Exception as e:
                    raise HTTPException(status_code=503, detail=f"Milvus 连接失败: {e}")
    return _mkv_client


# ---------- GET /api/v1/kb/docs ----------
class DocItem(BaseModel):
    id: str
    source: str
    pages: int
    preview: str
    chunk_count: int


@app.get("/api/v1/kb/docs")
def kb_docs(
    keyword: str = Query("", description="关键词过滤（匹配文件名）"),
    page: int = Query(1, ge=1, description="页码（从 1 开始）"),
    page_size: int = Query(20, ge=1, le=200, description="每页条数"),
):
    """按文档名聚合知识库块，支持关键词搜索与分页。"""
    mkv = _get_mkclient()
    # 先取所有 document 类型块
    rows = mkv.query(
        collection_name="rag_knowledge",
        filter='doc_type == "document"',
        output_fields=["text", "metadata"],
        limit=16384,
    )
    # 按 source 聚合
    from collections import defaultdict
    groups: Dict[str, List[Dict]] = defaultdict(list)
    for r in rows:
        meta = r.get("metadata") or {}
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except Exception:
                meta = {}
        src = meta.get("source", "unknown")
        groups[src].append({
            "page": meta.get("page", 1),
            "text": r.get("text", ""),
        })
    # 构建文档列表
    docs = []
    for src, chunks in groups.items():
        if keyword and keyword.lower() not in src.lower():
            continue
        pages = sorted({c["page"] for c in chunks})
        preview = chunks[0]["text"][:200] if chunks else ""
        docs.append(DocItem(
            id=src,
            source=src,
            pages=len(pages),
            preview=preview,
            chunk_count=len(chunks),
        ))
    # 分页
    total = len(docs)
    start = (page - 1) * page_size
    items = docs[start:start + page_size]
    return {
        "total": total,
        "page": page,
        "page_size": page_size,
        "items": [d.model_dump() for d in items],
    }


# ---------- GET /api/v1/kb/docs/{doc_id}/chunks ----------
@app.get("/api/v1/kb/docs/{doc_id}/chunks")
def kb_doc_chunks(
    doc_id: str,
    page: int = Query(1, ge=1, description="页码（从 1 开始）"),
    page_size: int = Query(20, ge=1, le=100, description="每页块数"),
):
    """获取指定文档的所有文本块，按页码排序，支持分页。"""
    mkv = _get_mkclient()
    rows = mkv.query(
        collection_name="rag_knowledge",
        filter=f'metadata["source"] == "{doc_id}" and doc_type == "document"',
        output_fields=["text", "metadata"],
        limit=16384,
    )
    parsed = []
    for r in rows:
        meta = r.get("metadata") or {}
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except Exception:
                meta = {}
        parsed.append({
            "page": meta.get("page", 1),
            "text": r.get("text", ""),
        })
    parsed.sort(key=lambda x: x["page"])
    total = len(parsed)
    start = (page - 1) * page_size
    items = parsed[start:start + page_size]
    return {
        "doc_id": doc_id,
        "total_pages": total,
        "page": page,
        "page_size": page_size,
        "total_chunks": total,
        "chunks": items,
    }


# ---------- GET /api/v1/kb/quadruples ----------
class QuadItem(BaseModel):
    id: str
    device: str
    fault_type: str
    feature_str: str
    solution: str
    page: Optional[int] = None


@app.get("/api/v1/kb/quadruples")
def kb_quadruples(
    fault_cn: str = Query("", description="故障类型过滤（留空查全部）"),
    device: str = Query("", description="设备过滤（留空查全部）"),
    keyword: str = Query("", description="关键词搜索（匹配故障特征/诊断方案）"),
    page: int = Query(1, ge=1, description="页码（从 1 开始）"),
    page_size: int = Query(20, ge=1, le=200, description="每页条数"),
):
    """查询四元组知识库（故障特征 + 诊断方案），支持多维度过滤与分页。"""
    mkv = _get_mkclient()
    # 先取所有 quadruple_text 块
    rows = mkv.query(
        collection_name="rag_knowledge",
        filter='doc_type == "quadruple_text"',
        output_fields=["text", "metadata"],
        limit=16384,
    )
    parsed = []
    for r in rows:
        text = r.get("text", "")
        # 解析文本: "设备：xxx。故障类型：xxx。故障特征向量：xxx。诊断方案：xxx"
        import re
        device_m = re.search(r"设备[：:]\s*([^。]+)", text)
        fault_m = re.search(r"故障类型[：:]\s*([^。]+)", text)
        feat_m = re.search(r"故障特征向量[：:]\s*([^。]+)", text)
        sol_m = re.search(r"诊断方案[：:]\s*(.+)", text, re.DOTALL)
        parsed.append(QuadItem(
            id=str(r.get("id", "")),
            device=device_m.group(1).strip() if device_m else "",
            fault_type=fault_m.group(1).strip() if fault_m else "",
            feature_str=feat_m.group(1).strip() if feat_m else "",
            solution=sol_m.group(1).strip() if sol_m else "",
            page=(r.get("metadata") or {}).get("page"),
        ))
    # 过滤
    filtered = parsed
    if fault_cn:
        filtered = [x for x in filtered if fault_cn.lower() in x.fault_type.lower()]
    if device:
        filtered = [x for x in filtered if device.lower() in x.device.lower()]
    if keyword:
        kw = keyword.lower()
        filtered = [x for x in filtered if kw in x.feature_str.lower() or kw in x.solution.lower()]
    total = len(filtered)
    start = (page - 1) * page_size
    items = filtered[start:start + page_size]
    return {
        "total": total,
        "page": page,
        "page_size": page_size,
        "items": [i.model_dump() for i in items],
    }


# ---------- GET /api/v1/diag/records ----------
@app.get("/api/v1/diag/records")
def diag_records(
    fault_cn: Optional[str] = Query(None, description="故障类型过滤（可选）"),
    days: Optional[int] = Query(None, description="最近 N 天（可选）"),
    page: int = Query(1, ge=1, description="页码（从 1 开始）"),
    page_size: int = Query(20, ge=1, le=200, description="每页条数"),
):
    """查询 MySQL 中的诊断记录（MySQL 未启用时返回 enabled=false）。"""
    if not MEMORY._mysql:
        return {
            "enabled": False,
            "hint": "MySQL 未启用，请设置环境变量 RAG_MYSQL_HOST 后重启服务",
            "total": 0,
            "page": page,
            "page_size": page_size,
            "items": [],
        }
    import time
    sql = "SELECT feat_hash, fault_cn, fault_desc, confidence, features, updated_at FROM rag_diag WHERE 1=1"
    args: List = []
    if fault_cn:
        sql += " AND fault_cn LIKE %s"
        args.append(f"%{fault_cn}%")
    if days:
        sql += " AND updated_at >= %s"
        args.append(time.time() - days * 86400)
    sql += " ORDER BY updated_at DESC LIMIT %s OFFSET %s"
    args.append(int(page_size))
    args.append((page - 1) * page_size)
    with MEMORY._mysql_lock:
        MEMORY._mysql.ping(reconnect=True)
        with MEMORY._mysql.cursor() as cur:
            cur.execute(sql, tuple(args))
            rows = cur.fetchall()
    # 总数查询（不含分页）
    cnt_sql = "SELECT COUNT(*) FROM rag_diag WHERE 1=1"
    cnt_args: List = []
    if fault_cn:
        cnt_sql += " AND fault_cn LIKE %s"
        cnt_args.append(f"%{fault_cn}%")
    if days:
        cnt_sql += " AND updated_at >= %s"
        cnt_args.append(time.time() - days * 86400)
    with MEMORY._mysql_lock:
        MEMORY._mysql.ping(reconnect=True)
        with MEMORY._mysql.cursor() as cur:
            cur.execute(cnt_sql, tuple(cnt_args))
            total = cur.fetchone()[0]
    items = []
    for row in rows:
        items.append({
            "feat_hash": row[0],
            "fault_cn": row[1],
            "fault_desc": row[2],
            "confidence": round(float(row[3]), 4) if row[3] else None,
            "features": json.loads(row[4]) if row[4] else None,
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(row[5])),
        })
    return {
        "enabled": True,
        "total": total,
        "page": page,
        "page_size": page_size,
        "items": items,
    }

