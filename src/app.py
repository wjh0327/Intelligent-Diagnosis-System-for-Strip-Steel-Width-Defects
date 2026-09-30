#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
带钢宽度缺陷诊断系统 - 自包含完整版
包含模型加载、检索、Agent 图构建，支持批量诊断结果持久化。
"""

import os
import json
import uuid
import csv
import hashlib
import io
import time
import warnings
from pathlib import Path
import pandas as pd

import streamlit as st

from service import MEMORY
from memory_store import MAX_STORED_MESSAGES
import service as _service_core
from logger import get_logger

warnings.filterwarnings("ignore")

# ==================== 长期记忆（Redis，自动降级到文件/内存） ====================
MEMORY = _service_core.MEMORY
logger = get_logger(__name__)

# ==================== 上传临时文件管理 ====================
TMP_UPLOAD_DIR = Path(__file__).resolve().parent.parent / "data" / "tmp_uploads"
TMP_FILE_MAX_AGE = 24 * 3600  # 超过 24 小时的临时上传文件自动清理
MAX_CSV_MB = 20               # CSV 上传上限：批量诊断按行耗 CPU，真实批量 574 行约 62KB


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


def _delete_tmp_upload(path) -> None:
    """删除 data/tmp_uploads/ 下的临时 CSV；路径防护：只删本目录内的文件。"""
    if not path:
        return
    try:
        _p = Path(path)
        if _p.is_file() and _p.resolve().is_relative_to(TMP_UPLOAD_DIR.resolve()):
            _p.unlink(missing_ok=True)
    except Exception as e:  # noqa: BLE001
        logger.warning("清理临时上传文件失败: %s", e)


@st.cache_data(show_spinner=False)
def _build_export_csv(batch_summary: str):
    """batch_summary JSON → (CSV 文本或 None, 导出行数, details 是否被截断)。

    以摘要字符串为缓存键：rerun 期间摘要不变则直接复用 CSV，
    不再每次 rerun 重新 json.loads + 重建；新批量诊断产生新摘要时自动换键。"""
    try:
        obj = json.loads(batch_summary)
        details = obj.get("details") or []
        if not details:
            return None, 0, False
        buf = io.StringIO()
        w = csv.writer(buf)
        # 特征列名取自 details 的 features_named（12维机理特征，FEAT12 列序）
        feat_cols = list(details[0].get("features_named", {}).keys())
        w.writerow(
            ["index", "stripno"] + feat_cols
            + ["fault_cn", "fault_desc", "confidence", "needs_review", "归因证据"]
        )
        for d in details:
            named = d.get("features_named") or {}
            w.writerow(
                [d.get("index", ""), d.get("stripno", "")]
                + [named.get(c, "") for c in feat_cols]
                + [d.get("fault_cn", ""), d.get("fault_desc", ""),
                   d.get("confidence", ""), d.get("needs_review", ""),
                   d.get("归因证据", "")]
            )
        return buf.getvalue(), len(details), bool(obj.get("details_truncated"))
    except Exception as e:  # noqa: BLE001
        logger.warning("导出诊断结果失败: %s", e)
        return None, 0, False


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
def init_resources():
    """模型/检索器/Agent 工具与图 —— 整个进程只构建一次。

    构建统一走 service.init_resources()（_init_lock + _RESOURCES 幂等，
    与 REST API 共用同一套初始化逻辑）；st.cache_resource 再包一层，
    保证 Streamlit 多会话、多次 rerun 不会重读 skills/*.md、重建
    ChatOpenAI / bind_tools 或重新编译 LangGraph 图。
    """
    try:
        return _service_core.init_resources()
    except Exception as e:  # noqa: BLE001
        st.error(f"初始化失败：{e}")
        st.stop()


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
        # 注意：变量不能命名为 csv——会遮蔽顶部 import csv 的模块，
        # 导致同一次 rerun 里导出区块的 csv.writer() 拿到字符串而失败
        restored_csv = restored_batch.get("csv_path") or ""
        if restored_csv and os.path.exists(restored_csv):
            st.session_state.csv_path = restored_csv
        elif st.session_state.batch_done or st.session_state.batch_summary:
            # 文件已不在（如停机超过 24h 被兜底清理）→ 批量状态随之作废并落盘，
            # 否则下次刷新又会从 MEMORY 复活"有摘要无文件"的陈旧状态，
            # Agent 会拿旧摘要回答"这个文件"的问题，而会话里根本没有文件
            st.session_state.batch_done = False
            st.session_state.batch_summary = ""
            MEMORY.save_batch(SID, {
                "batch_done": False,
                "batch_summary": "",
                "csv_path": "",
            })
            logger.warning(
                "会话 %s 的批量诊断临时文件已不存在（%s），批量状态已作废",
                SID[:8], restored_csv,
            )

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

# ==================== 历史诊断查询（MySQL 最近记录浏览） ====================
# 单卷追溯走对话：query_hist_diag_tool 按追溯键（STRIPNO+12维机理特征哈希）精确反查。
# 页面不提供 12 维特征手输查询：诊断记录的写入键含 STRIPNO（hybrid_model.strip_hash），
# 仅凭特征值无法构造出一致的哈希键，只能退化为 MySQL 全量浮点比对。
if MEMORY.status().get("mysql"):
    with st.expander("📚 最近诊断记录（MySQL）"):
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

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ==================== 批量诊断结果导出 ====================
# 位于对话渲染之后，与语义顺序一致；CSV 构建经 cache_data 缓存，rerun 不重建
if st.session_state.batch_done and st.session_state.batch_summary:
    csv_data, n_rows, truncated = _build_export_csv(st.session_state.batch_summary)
    if csv_data is not None:
        st.download_button(
            "📥 导出诊断结果 CSV",
            data=csv_data,
            file_name=f"诊断结果_{SID[:8]}.csv",
            mime="text/csv",
        )
        if truncated:
            st.caption(
                f"details 超过单次返回上限，导出仅含前 {n_rows} 条；"
                "完整逐卷结果已写入长期记忆，可在对话中按追溯键反查。"
            )

uploaded_file = st.file_uploader("📎 上传CSV文件（可选）", type="csv", key="csv_uploader")
if uploaded_file is not None and uploaded_file.size > MAX_CSV_MB * 1024 * 1024:
    st.error(
        f"CSV 文件 {uploaded_file.name} 为 {uploaded_file.size / 1024 / 1024:.1f} MB，"
        f"超过 {MAX_CSV_MB} MB 上限，已拒绝处理；批量诊断按行耗 CPU，超大文件请拆分。"
    )
elif uploaded_file is not None:
    # Streamlit 每次交互都会整体重跑本脚本，临时文件只在 CSV 真正变化时重新写盘
    # （uuid 路径每次重跑都不同，不能当变更指纹）。签名必须用内容哈希：
    # 现场反复导出的同名同大小 CSV（如只改一行等宽数值）靠名称+大小无法区分，
    # 会静默沿用旧临时文件，让用户拿到上一批的诊断结论。
    data = uploaded_file.getvalue()
    file_sig = hashlib.sha256(data).hexdigest()
    if st.session_state.get("csv_sig") != file_sig:
        # 换文件先删上一个临时文件：反复换 CSV 不再堆积（长驻进程等不到重启兜底清理）
        _delete_tmp_upload(st.session_state.get("csv_path"))
        TMP_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        tmp_path = TMP_UPLOAD_DIR / f"{uuid.uuid4().hex}.csv"
        tmp_path.write_bytes(data)
        st.session_state.csv_path = str(tmp_path)
        st.session_state.csv_sig = file_sig
        # 换了新 CSV → 旧批量诊断状态作废，Agent 会对新文件重新批量诊断
        # （与 REST /query 的 CSV 变更检测同语义）
        st.session_state.batch_done = False
        st.session_state.batch_summary = ""
        # 重置必须同步到持久层：浏览器刷新/WebSocket 重连会使 session_state 清空重建，
        # 恢复分支会从 MEMORY 拉回批量状态——若不清 Redis/文件里的旧状态，
        # 旧的 batch_done=True + 旧摘要 + 旧 csv_path 会覆盖本次重置
        MEMORY.save_batch(SID, {
            "batch_done": False,
            "batch_summary": "",
            "csv_path": str(tmp_path),
        })
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
                # 逐卷落库由 batch_csv_diagnosis_tool 内部全量完成（tools.py 的 save_diag），
                # 此处只更新会话内的批量摘要，不再重复写长期记忆
                st.session_state.batch_summary = new_summary
        except Exception as e:
            # 能走到这里的只有逃出 LangGraph 框架层的异常——LLM 超时与工具异常
            # 均已在 agent_node 内部转为错误文案。batch 状态保持调用前的一致
            # 快照即可（MEMORY 里本就是这些值），无需也不应在此另行同步；
            # 但必须留日志：这类失败在 app.log 里没有其它痕迹
            logger.exception("Agent invoke 失败: %s", e)
            answer = f"诊断过程出错：{str(e)}"
            steps = []

    if steps:
        step_text = "### 🧩 推理过程\n"
        for i, (tool, obs) in enumerate(steps, 1):
            obs_display = obs[:800] + "..." if len(obs) > 800 else obs
            step_text += f"\n**步骤 {i}: 调用 `{tool}`**\n\n```\n{obs_display}\n```\n"
        answer = step_text + "\n### 📋 最终诊断结论\n" + answer

    st.session_state.messages.append({"role": "assistant", "content": answer})
    # 本地与持久层同一上限：save_messages 落盘时截断到 MAX_STORED_MESSAGES，
    # 本地不裁剪会造成刷新恢复后"对话变短"的不一致，且推理过程（每步最多
    # 800 字符）随轮数累积，长会话的内存与 rerun 渲染负担持续增长
    if len(st.session_state.messages) > MAX_STORED_MESSAGES:
        st.session_state.messages = st.session_state.messages[-MAX_STORED_MESSAGES:]
    MEMORY.save_messages(SID, st.session_state.messages)
    MEMORY.save_batch(SID, {
        "batch_done": st.session_state.batch_done,
        "batch_summary": st.session_state.batch_summary,
        "csv_path": st.session_state.csv_path or "",
    })
    st.rerun()

if st.button("🗑️ 清除对话历史"):
    _delete_tmp_upload(st.session_state.csv_path)
    st.session_state.messages = []
    st.session_state.csv_path = None
    st.session_state.csv_sig = ""
    st.session_state.batch_done = False
    st.session_state.batch_summary = ""
    MEMORY.clear_session(SID)
    st.rerun()
