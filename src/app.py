#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
带钢宽度缺陷诊断系统 - 自包含完整版
包含模型加载、检索、Agent 图构建，支持批量诊断结果持久化。
"""

import os
import json
import uuid
import hashlib
import csv
import io
import time
import warnings
from pathlib import Path
import pandas as pd
from typing import List, Dict, Any, Generator, Tuple

import streamlit as st

from tools import create_tools
from agent_graph import build_agent_graph
from service import (
    DEEPSEEK_API_KEY,
    LLM_MODEL,
    MEMORY,
    SKILLS_DIR,
    batch_predict,
    cached_retrieve_text,
    load_classification_model as _service_load_classification_model,
    load_retriever as _service_load_retriever,
    predict_single,
    retrieve_context,
)
import service as _service_core
from logger import get_logger

warnings.filterwarnings("ignore")

# ==================== 长期记忆（Redis，自动降级到文件/内存） ====================
MEMORY = _service_core.MEMORY
logger = get_logger(__name__)

# ==================== 上传临时文件管理 ====================
TMP_UPLOAD_DIR = Path(__file__).resolve().parent.parent / "data" / "tmp_uploads"
TMP_FILE_MAX_AGE = 24 * 3600  # 超过 24 小时的临时上传文件自动清理


def _cleanup_tmp_uploads() -> None:
    """启动时清理过期临时上传文件，避免长期堆积。"""
    try:
        TMP_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        now = time.time()
        for f in TMP_UPLOAD_DIR.glob("*.csv"):
            try:
                if now - f.stat().st_mtime > TMP_FILE_MAX_AGE:
                    f.unlink(missing_ok=True)
                    logger.info("已清理过期临时文件: %s", f.name)
            except Exception:  # noqa: BLE001
                continue
    except Exception as e:  # noqa: BLE001
        logger.warning("临时文件清理失败: %s", e)


_cleanup_tmp_uploads()

FEATURE_DESC = {
    'FMWIDTHACTHOT': '精轧出口实际平均宽度热值',
    'FMWTARGETCOL': '精轧出口实际平均宽度冷值',
    'FMWTARGETHOT': '精轧出口目标平均宽度热值',
    'PDIWIDTHTOL': '带钢计划宽度余量',
    'RDWTARGETTOTAL': '粗轧出口目标平均宽度热值',
    'RMWIDTHACTHOT': '粗轧出口实际平均宽度热值',
    'is_DM': '卷取是否有波谷',
    'is_FM': '精轧是否有波谷',
    'is_HT': '精轧粗轧波谷位置是否同时在头部或尾部',
    'is_NA': '精轧和卷取宽度是否整体分布均匀且小于精轧目标宽度',
    'is_RM': '粗轧是否有波谷',
    'is_WS': '带钢中部宽度是否不均匀'
}

# ==================== 页面配置 ====================
st.set_page_config(
    page_title="带钢宽度缺陷诊断系统",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1E3A5F 0%, #2E5A88 100%);
        color: white;
        padding: 1.5rem 2rem;
        border-radius: 8px;
        margin-bottom: 2rem;
        display: flex;
        align-items: center;
        gap: 1rem;
    }
    .main-header h1 { color: white !important; margin: 0; }
    .main-header p { color: #B0C4DE; margin: 0.2rem 0 0 0; }
    .card { background: white; padding: 1.5rem; border-radius: 12px; box-shadow: 0 2px 12px rgba(0,0,0,0.08); margin-bottom: 1.5rem; }
    .stButton > button { background: linear-gradient(90deg, #2E5A88 0%, #1E3A5F 100%); color: white; border: none; border-radius: 6px; padding: 0.5rem 2rem; font-weight: bold; transition: all 0.2s; }
    .stButton > button:hover { background: linear-gradient(90deg, #1E3A5F 0%, #2E5A88 100%); box-shadow: 0 4px 12px rgba(46,90,136,0.4); }
    .success-box { background: #E6F7E9; border-left: 4px solid #2E7D32; padding: 1rem; border-radius: 4px; margin: 1rem 0; }
    .info-box { background: #E3F0FF; border-left: 4px solid #1E3A5F; padding: 1rem; border-radius: 4px; }
    .diagnosis-result { background: #FFF8E1; border-left: 4px solid #F57F17; padding: 1rem; border-radius: 4px; margin: 1rem 0; }
    .chat-container { border: 1px solid #ddd; border-radius: 10px; padding: 1rem; max-height: 400px; overflow-y: auto; background: #fafafa; margin-bottom: 1rem; }
    .chat-bubble { padding: 0.8rem; border-radius: 12px; margin: 0.5rem 0; max-width: 80%; }
    .chat-user { background: #DCF8C6; align-self: flex-end; }
    .chat-assistant { background: #E3F0FF; align-self: flex-start; }
    .stExpander { border: 1px solid #ddd; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# ==================== 核心逻辑复用 service 层（与 REST API 共享） ====================
@st.cache_resource(show_spinner=False)
def load_classification_model() -> Dict[str, Any]:
    try:
        return _service_load_classification_model()
    except RuntimeError as e:
        st.error(str(e))
        st.stop()


@st.cache_resource(show_spinner=False)
def load_retriever() -> Dict[str, Any]:
    return _service_load_retriever()


def init_resources():
    model_info = load_classification_model()
    retriever = load_retriever()
    _service_core._retriever_global = retriever
    feature_names = model_info['feature_names']
    tools = create_tools(
        model_info, retriever, feature_names,
        predict_single, retrieve_context,
        cached_retrieve_fn=cached_retrieve_text,
        batch_predict=batch_predict
    )
    agent_graph = build_agent_graph(
        tools,
        model_name=LLM_MODEL,
        api_key=DEEPSEEK_API_KEY,
        max_steps=6,
        skills_dir=SKILLS_DIR
    )
    return model_info, retriever, feature_names, agent_graph

model_info, retriever, feature_names, agent_graph = init_resources()

# ==================== 会话状态初始化（支持 Redis/文件长期记忆恢复） ====================
def _get_sid() -> str:
    """从 URL 查询参数获取稳定会话 ID，首次访问生成 32 位 hex。"""
    raw = st.query_params.get("sid")
    sid = raw[0] if isinstance(raw, list) and raw else (raw or "")
    if not MEMORY.is_valid_sid(sid):
        sid = uuid.uuid4().hex
        st.query_params["sid"] = sid
    return sid


SID = _get_sid()

if 'messages' not in st.session_state:
    restored = MEMORY.get_messages(SID)
    st.session_state.messages = restored if restored else []
if 'csv_path' not in st.session_state:
    st.session_state.csv_path = None
if 'batch_done' not in st.session_state:
    st.session_state.batch_done = False
if 'batch_summary' not in st.session_state:
    st.session_state.batch_summary = ""
    restored_batch = MEMORY.get_batch(SID)
    if restored_batch:
        st.session_state.batch_done = bool(restored_batch.get("batch_done", False))
        st.session_state.batch_summary = restored_batch.get("batch_summary", "")
        csv = restored_batch.get("csv_path") or ""
        if csv and os.path.exists(csv):
            st.session_state.csv_path = csv

# ==================== 主页面 ====================
st.markdown(
    """
    <div class="main-header">
        <span style="font-size: 2.5rem;">🔍</span>
        <div>
            <h1>带钢宽度缺陷智能诊断系统</h1>
            <p>输入问题、特征值或上传CSV，AI专家为您自主分析（支持多轮对话记忆）</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

mem_mode = {
    "redis": "🟢 Redis 长期记忆已启用",
    "file": "🟡 文件长期记忆已启用（Redis 未连接）",
    "memory": "⚪ 仅内存记忆（不跨重启）",
}.get(MEMORY.mode, MEMORY.mode)
st.caption(f"{mem_mode} · 会话ID：{SID[:8]}…")

# ==================== 历史诊断查询（跨会话追溯） ====================
with st.expander("🔍 历史诊断查询（按 12 维特征反查历史记录）"):
    hist_input = st.text_input("输入 12 维特征（逗号分隔）", key="hist_input")
    if st.button("查询历史诊断", key="hist_btn") and hist_input.strip():
        try:
            feat_list = [float(x.strip()) for x in hist_input.replace("，", ",").split(",")]
            if len(feat_list) != 12:
                st.warning("需要恰好 12 个特征值。")
            else:
                rec = MEMORY.get_diag_by_features(feat_list)
                if rec:
                    st.markdown(f"**故障类型**：{rec.get('fault_cn')}")
                    st.markdown(f"**故障描述**：{rec.get('fault_desc')}")
                    st.markdown(f"**置信度**：{rec.get('confidence')}")
                    st.json({"features": rec.get("features"), "updated_at": rec.get("updated_at")})
                else:
                    st.info("未找到该特征向量的历史诊断记录。")
        except Exception as e:  # noqa: BLE001
            st.error(f"输入解析失败：{e}")
    if MEMORY.status().get("mysql"):
        st.divider()
        st.markdown("**最近诊断记录（MySQL）**")
        hist_type = st.text_input("故障类型（可选，留空查全部）", key="hist_type")
        hist_days = st.number_input(
            "最近天数", min_value=1, max_value=365, value=7, key="hist_days"
        )
        if st.button("查询最近诊断记录", key="hist_recent_btn"):
            recs = MEMORY.get_diag_history(
                fault_cn=hist_type.strip() or None,
                days=int(hist_days),
                limit=50,
            )
            if recs:
                lines = []
                for r in recs:
                    ts = time.strftime(
                        "%Y-%m-%d %H:%M", time.localtime(r.get("updated_at") or 0)
                    )
                    lines.append(
                        f"- **{r.get('fault_cn')}**（置信度 {r.get('confidence')}，{ts}）："
                        f"{r.get('fault_desc')}"
                    )
                st.markdown("\n".join(lines))
            else:
                st.info("没有符合条件的诊断记录。")

# ==================== 批量诊断结果导出 ====================
if st.session_state.batch_done and st.session_state.batch_summary:
    try:
        _bobj = json.loads(st.session_state.batch_summary)
        _details = _bobj.get("details") or []
        if _details:
            _buf = io.StringIO()
            _w = csv.writer(_buf)
            _w.writerow(
                ["index"] + [f"f{i+1}" for i in range(12)]
                + ["fault_cn", "fault_desc", "confidence"]
            )
            for _d in _details:
                _feats = list(_d.get("features", []) or [])
                _w.writerow(
                    [_d.get("index", "")]
                    + _feats
                    + [_d.get("fault_cn", ""), _d.get("fault_desc", ""), _d.get("confidence", "")]
                )
            st.download_button(
                "📥 导出诊断结果 CSV",
                data=_buf.getvalue(),
                file_name=f"诊断结果_{SID[:8]}.csv",
                mime="text/csv",
            )
    except Exception as e:  # noqa: BLE001
        logger.warning("导出诊断结果失败: %s", e)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

uploaded_file = st.file_uploader("📎 上传CSV文件（可选）", type="csv", key="csv_uploader")
if uploaded_file is not None:
    TMP_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    tmp_path = TMP_UPLOAD_DIR / f"{uuid.uuid4().hex}.csv"
    tmp_path.write_bytes(uploaded_file.getvalue())
    st.session_state.csv_path = str(tmp_path)
    st.info(f"已上传文件：{uploaded_file.name}，后续对话将默认使用该文件。")
elif st.session_state.csv_path:
    st.info(f"当前使用的文件：{Path(st.session_state.csv_path).name}")

prompt = st.chat_input("请输入您的问题（可包含12个特征值，或上传CSV后进行提问）...")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    MAX_HISTORY_ROUNDS = 4
    history_limit = MAX_HISTORY_ROUNDS * 2
    recent_history = st.session_state.messages[:-1][-history_limit:]
    csv_path_to_use = ""
    if st.session_state.csv_path and not st.session_state.batch_done:
        csv_path_to_use = st.session_state.csv_path

    initial_state = {
        "query": prompt,
        "features": "",
        "csv_path": csv_path_to_use,
        "conversation_history": recent_history,
        "intermediate_steps": [],
        "final_answer": "",
        "step_count": 0,
        "batch_done": st.session_state.batch_done,
        "batch_summary": st.session_state.batch_summary,
    }

    with st.spinner("专家正在分析，请稍候..."):
        try:
            final_state = agent_graph.invoke(initial_state)
            steps = final_state.get("intermediate_steps", [])
            answer = final_state.get("final_answer", "未能生成回答。")
            st.session_state.batch_done = final_state.get("batch_done", False)
            new_summary = final_state.get("batch_summary", "")
            if new_summary:
                st.session_state.batch_summary = new_summary
                # 批量诊断详情按特征哈希写入长期记忆，便于后续跨会话追溯
                try:
                    batch_obj = json.loads(new_summary)
                    if isinstance(batch_obj, dict) and batch_obj.get("details"):
                        for d in batch_obj["details"]:
                            feats = d.get("features")
                            if feats:
                                feat_hash = hashlib.md5(
                                    json.dumps(feats, ensure_ascii=False).encode("utf-8")
                                ).hexdigest()
                                MEMORY.save_diag(feat_hash, {
                                    "fault_cn": d.get("fault_cn"),
                                    "fault_desc": d.get("fault_desc"),
                                    "confidence": d.get("confidence"),
                                    "features": feats,
                                })
                except Exception:
                    pass
        except Exception as e:
            answer = f"诊断过程出错：{str(e)}"
            steps = []

    if steps:
        step_text = "### 🧩 推理过程\n"
        for i, (tool, obs) in enumerate(steps, 1):
            obs_display = obs[:800] + "..." if len(obs) > 800 else obs
            step_text += f"\n**步骤 {i}: 调用 `{tool}`**\n\n```\n{obs_display}\n```\n"
        answer = step_text + "\n### 📋 最终诊断结论\n" + answer

    st.session_state.messages.append({"role": "assistant", "content": answer})
    MEMORY.save_messages(SID, st.session_state.messages)
    MEMORY.save_batch(SID, {
        "batch_done": st.session_state.batch_done,
        "batch_summary": st.session_state.batch_summary,
        "csv_path": st.session_state.csv_path or "",
    })
    st.rerun()

if st.button("🗑️ 清除对话历史"):
    if st.session_state.csv_path:
        try:
            _p = Path(st.session_state.csv_path)
            if _p.is_file() and str(_p).startswith(str(TMP_UPLOAD_DIR)):
                _p.unlink(missing_ok=True)
        except Exception:  # noqa: BLE001
            pass
    st.session_state.messages = []
    st.session_state.csv_path = None
    st.session_state.batch_done = False
    st.session_state.batch_summary = ""
    MEMORY.clear_session(SID)
    st.rerun()
