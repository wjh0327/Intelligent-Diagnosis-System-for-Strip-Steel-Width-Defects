# -*- coding: utf-8 -*-
"""
src/api.py —— REST API 服务
============================
供 MES/ERP 等外部系统集成调用诊断能力：

  GET  /api/health                健康检查
  GET  /api/v1/meta/features      特征字段元信息（顺序、含义）
  POST /api/v1/diagnose           单条诊断（12 维特征向量）
  POST /api/v1/diagnose/batch     批量诊断（JSON 数组）
  POST /api/v1/diagnose/batch/file 批量诊断（CSV 文件上传）
  POST /api/v1/query              LLM 知识问答/Agent 诊断（需 DEEPSEEK_API_KEY）

启动：python -m uvicorn src.api:app --host 0.0.0.0 --port 8000
"""

import io
import sys
import threading
import uuid
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

# 兼容 `python -m uvicorn src.api:app` 与直接运行两种方式
sys.path.insert(0, str(Path(__file__).resolve().parent))

from logger import get_logger  # noqa: E402
from service import (
    DEEPSEEK_API_KEY,
    MEMORY,
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
    features: List[float] = Field(
        ..., min_length=12, max_length=12,
        description="按模型特征顺序排列的 12 维特征向量",
    )


class BatchDiagnoseRequest(BaseModel):
    samples: List[List[float]] = Field(
        ..., min_length=1,
        description="多个 12 维特征向量组成的数组",
    )


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    features: Optional[str] = Field("", description="逗号分隔的 12 维特征值（可选）")
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
    loaded = _resources is not None
    return {
        "status": "ok",
        "service": "steel-width-defect-diagnosis",
        "resources_loaded": loaded,
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


@app.post("/api/v1/diagnose")
def diagnose(req: DiagnoseRequest):
    model_info, _r, _f, _a = get_resources()
    try:
        sample_df = pd.DataFrame([req.features], columns=model_info['feature_names'])
        diag = predict_single(model_info, sample_df)
    except Exception as e:  # noqa: BLE001
        logger.error("单条诊断失败: %s", e)
        raise HTTPException(status_code=500, detail=f"诊断失败: {e}")
    probs = diag.get("probs") or []
    confidence = round(max(probs), 6) if probs else None
    imp = model_info.get('importance_cache', {}).get(diag['fault_cn'], [])[:10]
    return {
        "request_id": uuid.uuid4().hex[:12],
        "fault_cn": diag['fault_cn'],
        "fault_desc": diag['fault_desc'],
        "confidence": confidence,
        "probabilities": _prob_map(model_info, probs),
        "top_features": [
            {"feature": f, "importance": round(float(w), 6)} for f, w in imp
        ],
    }


def _run_batch(samples: List[List[float]]):
    model_info, _r, _f, _a = get_resources()
    df = pd.DataFrame(samples, columns=model_info['feature_names'])
    results = batch_predict(model_info, df)
    distribution: Dict[str, int] = {}
    out = []
    for i, r in enumerate(results):
        fault_cn = r['fault_cn']
        distribution[fault_cn] = distribution.get(fault_cn, 0) + 1
        probs = r.get("probs") or []
        out.append({
            "index": i,
            "fault_cn": fault_cn,
            "fault_desc": r['fault_desc'],
            "confidence": round(max(probs), 6) if probs else None,
            "features": samples[i],
        })
    return {
        "request_id": uuid.uuid4().hex[:12],
        "total": len(out),
        "distribution": distribution,
        "results": out,
    }


@app.post("/api/v1/diagnose/batch")
def diagnose_batch(req: BatchDiagnoseRequest):
    try:
        return _run_batch(req.samples)
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
    samples = df[model_info['feature_names']].values.astype(float).tolist()
    return _run_batch(samples)


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

    _m, _r, _f, agent_graph = get_resources()
    initial_state = {
        "query": req.question,
        "features": req.features or "",
        "csv_path": "",
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
            "csv_path": "",
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
        "history_length": len(history) if sid else len(req.conversation_history or []),
    }
