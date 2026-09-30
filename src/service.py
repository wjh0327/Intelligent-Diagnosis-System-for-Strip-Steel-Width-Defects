# -*- coding: utf-8 -*-
"""
src/service.py —— Streamlit 无关的核心服务层
==============================================
封装模型加载、故障分类、知识检索、相似案例与 Agent 图构建，
供 REST API（api.py）与后续外部系统（MES/ERP）复用。

注意：本模块逻辑与 app.py 保持一致；app.py 额外承担 Streamlit 交互层职责。
"""

import hashlib
import json
import os
import threading
import time
import warnings
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import torch
from pymilvus import MilvusClient
from sentence_transformers import CrossEncoder, SentenceTransformer

from agent_graph import build_agent_graph
from hybrid_model import (FEAT_CN as _FEAT_CN,
                          LABEL_MAPPING as _LABEL_MAPPING,
                          REQUIRED_COLUMNS as _REQUIRED_COLUMNS,
                          load_model as _load_hybrid_model,
                          predict_strips as _hybrid_predict_strips,
                          scale_feat12 as _scale_feat12,
                          strip_hash as _strip_hash)
from kb_utils import is_useful_chunk
from logger import get_logger
from memory_store import get_memory_store
from tools import create_tools

warnings.filterwarnings("ignore")

# ==================== 全局路径配置 ====================
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
MILVUS_DB = str(PROJECT_ROOT / "milvus_kb.db")  # Milvus Lite 数据文件（项目目录含中文时可能无法读写）
EMBED_MODEL_PATH = str(PROJECT_ROOT / "models" / "bge-large-zh-v1.5")
RERANKER_MODEL_PATH = str(PROJECT_ROOT / "models" / "bge-reranker-v2-m3")
SKILLS_DIR = str(PROJECT_ROOT / "skills")

LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-flash").strip() or "deepseek-flash"
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")

TOP_K_RETRIEVAL = 5
TOP_K_RERANK = 6  # 重排后注入 LLM 的知识块数；每类 doc_type 至少保底 1 条
SIM_CASE_LIMIT = 5
MAX_CONTEXT_CHARS = 4000  # 注入 LLM 的检索上下文长度上限

# 检索配置指纹：进入检索缓存键（见 cached_retrieve_text）。嵌入模型目录、
# 重排器目录存在与否、关键检索参数任一变化都会换键，避免旧配置的结果
# 被新配置命中。注意：只能感知目录名，模型文件原位替换无法检测。
_RETRIEVE_SIG = hashlib.sha1(json.dumps({
    "embed": Path(EMBED_MODEL_PATH).name,
    "rerank": Path(RERANKER_MODEL_PATH).name if os.path.exists(RERANKER_MODEL_PATH) else None,
    "top_k": TOP_K_RETRIEVAL,
    "top_rerank": TOP_K_RERANK,
    "max_ctx": MAX_CONTEXT_CHARS,
}, sort_keys=True).encode("utf-8")).hexdigest()[:8]

# ==================== 长期记忆（Redis，自动降级到文件/内存） ====================
MEMORY = get_memory_store()
logger = get_logger(__name__)


def load_classification_model() -> Dict[str, Any]:
    """加载论文主模型（双分支混合CNN，单种子）。

    返回 model_info 供工具层与 API 使用：
    - feature_names: 诊断必需的原始带钢列（6工艺参数 + 3段全长宽度曲线；STRIPNO/label 可选）
    - importance_cache: 按故障类型预计算的梯度归因 Top10（中文名，训练集按类聚合）
    - scale_feat12 / strip_hash: 12维特征标准化与历史追溯键，由 hybrid_model 提供
    """
    state = _load_hybrid_model()
    importance_cache: Dict[str, List[Tuple[str, float]]] = {
        fault: [(_FEAT_CN.get(n, n), float(w)) for n, w in pairs[:10]]
        for fault, pairs in state['importance'].items()
    }
    return {
        'feature_names': list(_REQUIRED_COLUMNS),
        'label_mapping': dict(_LABEL_MAPPING),
        'importance_cache': importance_cache,
        'scale_feat12': _scale_feat12,
        'strip_hash': _strip_hash,
    }


def load_retriever() -> Dict[str, Any]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embed = SentenceTransformer(EMBED_MODEL_PATH, device=device)
    embed.max_seq_length = 128
    client = MilvusClient(MILVUS_DB)
    client.load_collection("rag_knowledge")
    client.load_collection("fault_feature_vectors")
    reranker = None
    reranker_degraded = False
    if os.path.exists(RERANKER_MODEL_PATH):
        try:
            reranker = CrossEncoder(RERANKER_MODEL_PATH, max_length=512, device=device)
        except Exception as e:
            # 配置了重排器但加载失败：检索退化为未重排结果。不能静默——
            # 降级结果不入检索缓存（见 cached_retrieve_text），靠此标记识别
            reranker_degraded = True
            logger.warning("重排器加载失败，检索将退化为未重排结果: %s", e)
    return {'embed': embed, 'client': client, 'reranker': reranker,
            'reranker_degraded': reranker_degraded}


# ==================== 模型预测 ====================
def batch_predict(model_info: Dict, df: pd.DataFrame, batch_size: int = 64) -> List[Dict]:
    """对原始带钢记录 DataFrame 批量预测（曲线分支 ＋ 12维机理特征分支）。

    df 须含 model_info['feature_names'] 所列原始列（STRIPNO/label 可选；
    6个 is_* 规则标志位由 hybrid_model 按 te.py 原版逻辑现算，无需提供）。
    数据缺失的行标记为'数据不完整'并转人工复核，不参与预测。"""
    return _hybrid_predict_strips(df, with_evidence=True)


def predict_single(model_info: Dict, sample_df: pd.DataFrame) -> Dict:
    return batch_predict(model_info, sample_df)[0]


# ---- 相似案例缓存（按向量库版本戳校验，随知识库更新自动失效） ----
# 不能使用 functools.lru_cache：它无法感知 fault_feature_vectors 集合的重建/追加，
# 而知识库管理页（app_file_uploader）与 rebuild_feature_vectors.py 都会改动该集合。
# 这里用「版本戳 + 有界字典」替代，版本戳由两个来源合成：
#   1. _vector_cache_version —— 同进程的失效回调把它 +1（见 invalidate_inprocess_caches）；
#   2. fault_feature_vectors 集合目录的文件时间戳 —— 覆盖「另一个进程改动了集合」的场景。
# 注意：第 2 项必须看集合自己的目录，不能看 milvus_kb.db 根目录——Milvus Lite 把某个
# 集合的数据写在 collections/<集合名>/ 两层之下，根目录 mtime 只在直接子项增删时才变，
# 集合重建/追加都不会动它（实测根目录停留在 2026-08-15，而集合在 9-26 被重建）。
_VECTOR_CACHE_MAX = 128
_VECTOR_STAMP_TTL = 5.0   # 最多每 5 秒扫一次集合目录，避免每请求都读盘
_VECTOR_COLLECTION_DIR = "fault_feature_vectors"
_VECTOR_VERSION_LOCK = threading.Lock()
_vector_cache: "OrderedDict[Tuple[int, float, Tuple[float, ...]], str]" = OrderedDict()
_vector_cache_version = 0
# (最近扫描时刻, 最近扫描得到的集合目录时间戳)
_vector_stamp_cache: Tuple[float, float] = (0.0, -1.0)


def _store_vector_dir_stamp(checked_at: float, stamp: float) -> None:
    """记录本次集合目录扫描结果（单独成函数以便正确声明 global）。"""
    global _vector_stamp_cache
    _vector_stamp_cache = (checked_at, stamp)


def _vector_dir_stamp() -> float:
    """fault_feature_vectors 集合目录下所有文件的最大 mtime；目录不存在返回 0。

    取该集合目录下**全部**文件（manifest.json / *.parquet / *.idx 等）的最大值，
    而不是只看 manifest：Milvus Lite 的 DML 追加会写 WAL 与数据文件，manifest 只在
    部分操作（DDL / flush）时重写，只看 manifest 会漏掉纯追加。该集合文件数很少
    （当前 6 个），扫描成本可忽略。

    跨进程（另一个 Streamlit 进程 / rebuild_feature_vectors.py）改动集合后，此值必然
    变化，从而让本进程的相似案例缓存自动失效——这是进程内回调无法覆盖的场景。
    """
    now = time.time()
    checked_at, stamp = _vector_stamp_cache
    if now - checked_at < _VECTOR_STAMP_TTL:
        return stamp
    coll_dir = os.path.join(MILVUS_DB, "collections", _VECTOR_COLLECTION_DIR)
    newest = 0.0
    try:
        for root, _dirs, files in os.walk(coll_dir):
            for name in files:
                if name.endswith(".prev"):   # manifest 轮转备份，不代表新写入
                    continue
                try:
                    mtime = os.path.getmtime(os.path.join(root, name))
                except OSError:
                    continue
                if mtime > newest:
                    newest = mtime
    except OSError:
        newest = 0.0
    _store_vector_dir_stamp(now, newest)
    return newest


def _vector_cache_stamp() -> Tuple[int, float]:
    """当前向量库版本戳；跨进程改动通过集合目录的文件时间戳体现。"""
    return (_vector_cache_version, _vector_dir_stamp())


def _search_similar(stamp: Tuple[int, float], feat_vec_tuple: Tuple[float, ...]) -> str:
    """按标准化 12 维特征查相似历史案例（带向量库版本戳的进程内缓存）。

    缓存键含版本戳，fault_feature_vectors 被追加/重建后自动失效：
    同进程的改动由 memory_store.invalidate_retrieve_cache() 触发的回调覆盖，
    其他进程的改动由 _vector_dir_stamp() 的集合目录文件时间戳覆盖。
    检索器统一取全局单例。
    """
    key = (stamp[0], stamp[1], feat_vec_tuple)

    with _VECTOR_VERSION_LOCK:
        hit = _vector_cache.get(key)
        if hit is not None:
            _vector_cache.move_to_end(key)
            return hit

    client = _get_retriever()["client"]
    feat_list = list(feat_vec_tuple)
    sim_res = client.search(
        collection_name="fault_feature_vectors", data=[feat_list],
        limit=SIM_CASE_LIMIT, output_fields=["label", "plan"],
        search_params={"metric_type": "COSINE", "params": {"nprobe": 8}},
    )[0]
    if not sim_res:
        # 空结果不入缓存：集合刚重建时可能暂时不可见，避免把空结果钉住
        return ""
    # Milvus COSINE 此处返回的是余弦距离(1−cos，越小越相似)，转成相似度展示
    sim_lines = [
        f"故障类型：{h['entity']['label']}，相似度：{1.0 - float(h['distance']):.4f}，"
        f"方案：{h['entity']['plan']}"
        for h in sim_res
    ]
    sim_text = "【相似历史案例】\n" + "\n".join(sim_lines)

    with _VECTOR_VERSION_LOCK:
        _vector_cache[key] = sim_text
        _vector_cache.move_to_end(key)
        while len(_vector_cache) > _VECTOR_CACHE_MAX:
            _vector_cache.popitem(last=False)
    return sim_text


# ==================== 知识检索（含缓存） ====================
def retrieve_context(retriever, query, feat_vec=None):
    embed = retriever['embed']
    client = retriever['client']
    reranker = retriever['reranker']
    if not query and feat_vec is None:
        return "", ""
    if query:
        query_vec = embed.encode(query, normalize_embeddings=True).tolist()
    else:
        query_vec = None
    candidates = []
    if query_vec:
        doc_types = ["document", "quadruple_text", "quadruple_kw"]

        def search_doc_type(dt):
            return client.search(
                collection_name="rag_knowledge", data=[query_vec],
                filter=f'doc_type == "{dt}"', limit=TOP_K_RETRIEVAL,
                output_fields=["text", "doc_type"],
                search_params={"metric_type": "COSINE", "params": {"nprobe": 8}},
            )[0]

        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {executor.submit(search_doc_type, dt): dt for dt in doc_types}
            for future in as_completed(futures):
                for h in future.result():
                    text = h['entity']['text']
                    if not is_useful_chunk(text):
                        continue
                    candidates.append({
                        'text': text,
                        'doc_type': h['entity'].get('doc_type', ''),
                        'score': h['distance'],
                    })
    def _pick_balanced(cands, limit):
        """保证每类 doc_type 至少 1 条，再按序补满。"""
        selected = []
        picked = set()
        for dt in ["document", "quadruple_text", "quadruple_kw"]:
            for c in cands:
                if id(c) not in picked and c.get('doc_type') == dt:
                    selected.append(c)
                    picked.add(id(c))
                    break
        for c in cands:
            if len(selected) >= limit:
                break
            if id(c) not in picked:
                selected.append(c)
                picked.add(id(c))
        return selected[:limit]

    if candidates:
        if reranker:
            candidates = candidates[:15]
            pairs = [(query if query else "", c['text']) for c in candidates]
            scores = reranker.predict(pairs)
            for c, s in zip(candidates, scores):
                c['rerank_score'] = float(s)
            candidates.sort(key=lambda x: x.get('rerank_score', 0), reverse=True)
        candidates = _pick_balanced(candidates, TOP_K_RERANK)
    top_blocks = candidates
    context = "\n\n".join(
        f"[{i+1}] ({c['doc_type']}) {c['text']}" for i, c in enumerate(top_blocks)
    )
    def truncate_context(ctx, max_chars):
        if len(ctx) <= max_chars:
            return ctx
        truncated = ctx[:max_chars]
        last_newline = truncated.rfind('\n')
        if last_newline > max_chars // 2:
            return truncated[:last_newline]
        return truncated

    context = truncate_context(context, MAX_CONTEXT_CHARS)

    sim_text = ""
    if feat_vec and len(feat_vec) == 12:
        # 版本戳在此取一次，使「查库」与「写缓存」共用同一版本，避免失效竞态
        sim_text = _search_similar(_vector_cache_stamp(), tuple(feat_vec))
    return context, sim_text


# ==================== 检索结果缓存（不缓存空结果） ====================
# 进程内 LRU 兜底缓存：仅在 MEMORY 未启用（Redis/文件后端均不可用）时读写，
# 与 MEMORY 缓存路径互斥。条目带写入时间戳，受 TTL + 容量双重约束，
# 与 Redis/文件路径的 24 小时语义一致，避免热键被反复命中就永不过期。
_RETRIEVE_CACHE_MAX = 512
_retrieve_cache: "OrderedDict[Tuple[str, str], Tuple[float, Tuple[str, str]]]" = OrderedDict()
_RETRIEVE_CACHE_TTL = 60 * 60 * 24  # 24 小时（两条路径共用）

# ==================== 检索缓存统一失效（供知识库更新端调用） ====================
def invalidate_inprocess_caches() -> int:
    """清空本进程的全部检索类缓存，返回清空的条目数。

    - _retrieve_cache：未启用 Redis/文件后端时的文本检索 LRU；
    - _vector_cache：相似案例缓存（另有集合目录时间戳作双保险）。

    本函数在模块末尾注册到 memory_store.register_invalidate_callback()，
    因此知识库更新端只需调用 memory_store.invalidate_retrieve_cache()，
    本进程的两级缓存就会随之失效，无需各自记忆要清哪些缓存。
    """
    global _vector_cache_version
    with _VECTOR_VERSION_LOCK:
        n = len(_retrieve_cache) + len(_vector_cache)
        _retrieve_cache.clear()
        _vector_cache.clear()
        # 版本戳递增：即便有并发请求正持旧版本戳写回，也不会被后续读命中
        _vector_cache_version += 1
    return n


def cached_retrieve_text(query: str) -> Tuple[str, str]:
    """带缓存的检索，但跳过空结果的缓存，避免空结果被永久命中。"""
    key = query
    cache_key = "rag:cache:retrieve:" + hashlib.sha1(
        json.dumps([query, _RETRIEVE_SIG], ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    if MEMORY.enabled:
        cached = MEMORY.cache_get(cache_key)
        if cached and isinstance(cached, list) and len(cached) == 2:
            return tuple(cached)
    else:
        hit = _retrieve_cache.get(key)
        if hit is not None:
            stored_at, value = hit
            if time.time() - stored_at <= _RETRIEVE_CACHE_TTL:
                _retrieve_cache.move_to_end(key)
                return value
            _retrieve_cache.pop(key, None)  # 过期立即逐出，不等 LRU 淘汰

    context, sim_text = retrieve_context(_get_retriever(), query)
    # 重排器配置了但加载失败时结果为未重排的降级质量，不入任何缓存——
    # Redis 键跨重启存活，否则重排器恢复后同一键仍会命中降级结果
    degraded = bool(_get_retriever().get("reranker_degraded"))
    if context and not degraded:
        if MEMORY.enabled:
            MEMORY.cache_set(cache_key, list([context, sim_text]), ttl=_RETRIEVE_CACHE_TTL)
        else:
            _retrieve_cache[key] = (time.time(), (context, sim_text))
            _retrieve_cache.move_to_end(key)
            while len(_retrieve_cache) > _RETRIEVE_CACHE_MAX:
                _retrieve_cache.popitem(last=False)
    return context, sim_text


_retriever_global: Optional[Dict[str, Any]] = None


def _get_retriever() -> Dict[str, Any]:
    """返回全局检索器；init_resources 已加载则直接复用，否则惰性加载。"""
    global _retriever_global
    if _retriever_global is None:
        _retriever_global = load_retriever()
    return _retriever_global


# ==================== 资源初始化（供 API 复用） ====================
_RESOURCES: Optional[Tuple[Any, Any, List[str], Any]] = None
_init_lock = threading.Lock()


def init_resources():
    """线程安全且幂等的资源初始化：模型/检索器/Agent 仅在首次调用时加载一次。"""
    global _RESOURCES, _retriever_global
    if _RESOURCES is not None:
        return _RESOURCES
    with _init_lock:
        if _RESOURCES is None:
            logger.info("开始初始化模型与检索资源...")
            model_info = load_classification_model()
            retriever = load_retriever()
            _retriever_global = retriever
            feature_names = model_info['feature_names']
            tools = create_tools(
                model_info, retriever, feature_names,
                predict_single, retrieve_context,
                cached_retrieve_fn=cached_retrieve_text,
                batch_predict=batch_predict,
            )
            agent_graph = build_agent_graph(
                tools,
                model_name=LLM_MODEL,
                api_key=DEEPSEEK_API_KEY,
                max_steps=6,
                skills_dir=SKILLS_DIR,
            )
            _RESOURCES = (model_info, retriever, feature_names, agent_graph)
            logger.info("模型与检索资源初始化完成")
    return _RESOURCES


# ==================== 检索缓存失效接线 ====================
# 知识库写入端（app_file_uploader / rebuild_feature_vectors）统一调用
# memory_store.get_memory_store().invalidate_retrieve_cache()；在此把它接到
# 本进程的缓存清理上，使 Redis/文件后端与 _retrieve_cache / _vector_cache 同步失效。
# 边界：回调只能覆盖「同一进程」的失效。另一个进程（例如独立运行的
# app_file_uploader）改了 fault_feature_vectors 时，本进程由 _vector_dir_stamp()
# 的集合目录时间戳感知——两条路径合起来才是完整的。
MEMORY.register_invalidate_callback(invalidate_inprocess_caches)
