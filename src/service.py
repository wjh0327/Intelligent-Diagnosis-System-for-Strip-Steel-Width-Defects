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
import re
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from pymilvus import MilvusClient
from sentence_transformers import CrossEncoder, SentenceTransformer

from agent_graph import build_agent_graph
from hybrid_model import (FEAT_CN as _FEAT_CN,
                          LABEL_MAPPING as _LABEL_MAPPING,
                          PROBLEM_DESCRIPTIONS as _PROBLEM_DESCRIPTIONS,
                          REQUIRED_COLUMNS as _REQUIRED_COLUMNS,
                          load_model as _load_hybrid_model,
                          predict_strips as _hybrid_predict_strips,
                          record_to_feat12 as _record_to_feat12,
                          scale_feat12 as _scale_feat12,
                          strip_hash as _strip_hash)
from logger import get_logger
from memory_store import get_memory_store
from tools import create_tools

warnings.filterwarnings("ignore")

# ==================== 全局路径配置 ====================
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
MODEL_DIR = str(PROJECT_ROOT / "models")
MILVUS_DB = str(PROJECT_ROOT / "milvus_kb.db")  # Milvus Lite 数据文件（项目目录含中文时可能无法读写）
EMBED_MODEL_PATH = str(PROJECT_ROOT / "models" / "bge-large-zh-v1.5")
RERANKER_MODEL_PATH = str(PROJECT_ROOT / "models" / "bge-reranker-v2-m3")
SKILLS_DIR = str(PROJECT_ROOT / "skills")

LLM_MODEL = "deepseek-v4-flash"  # DeepSeek 官方模型
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")

TOP_K_RETRIEVAL = 5
TOP_K_RERANK = 6  # 重排后注入 LLM 的知识块数；每类 doc_type 至少保底 1 条
SIM_CASE_LIMIT = 5
MAX_LLM_WORKERS = 5

# ==================== 长期记忆（Redis，自动降级到文件/内存） ====================
MEMORY = get_memory_store()
logger = get_logger(__name__)


def load_classification_model() -> Dict[str, Any]:
    """加载论文主模型（双分支混合CNN，单种子，替代原 MIC 注意力集成模型）。

    返回 model_info 供工具层与 API 使用：
    - feature_names: 诊断必需的原始带钢列（6工艺参数 + 3段全长宽度曲线；STRIPNO/label 可选）
    - importance_cache: 按故障类型预计算的梯度归因 Top10（中文名，训练集按类聚合）
    - 12维特征的标准化与相似检索/历史哈希辅助函数由 hybrid_model 提供，
      经 model_info['record_to_feat12'/'scale_feat12'/'strip_hash'] 暴露给工具层。
    """
    state = _load_hybrid_model()
    importance_cache: Dict[str, List[Tuple[str, float]]] = {
        fault: [(_FEAT_CN.get(n, n), float(w)) for n, w in pairs[:10]]
        for fault, pairs in state['importance'].items()
    }
    return {
        'model': state['model'],
        'ckpt': state['ckpt'],
        'feature_names': list(_REQUIRED_COLUMNS),
        'label_mapping': dict(_LABEL_MAPPING),
        'problem_descriptions': dict(_PROBLEM_DESCRIPTIONS),
        'importance_cache': importance_cache,
        'record_to_feat12': _record_to_feat12,
        'scale_feat12': _scale_feat12,
        'strip_hash': _strip_hash,
        'device': 'cpu',
    }


def load_retriever() -> Dict[str, Any]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embed = SentenceTransformer(EMBED_MODEL_PATH, device=device)
    embed.max_seq_length = 128
    client = MilvusClient(MILVUS_DB)
    client.load_collection("rag_knowledge")
    client.load_collection("fault_feature_vectors")
    reranker = None
    if os.path.exists(RERANKER_MODEL_PATH):
        try:
            reranker = CrossEncoder(RERANKER_MODEL_PATH, max_length=512, device=device)
        except Exception:
            pass
    return {'embed': embed, 'client': client, 'reranker': reranker}


# ==================== 模型预测 ====================
def batch_predict(model_info: Dict, df: pd.DataFrame, batch_size: int = 64) -> List[Dict]:
    """对原始带钢记录 DataFrame 批量预测（曲线分支 ＋ 12维机理特征分支）。

    df 须含 model_info['feature_names'] 所列原始列（STRIPNO/label 可选；
    6个 is_* 规则标志位由 hybrid_model 按 te.py 原版逻辑现算，无需提供）。
    数据缺失的行标记为'数据不完整'并转人工复核，不参与预测。"""
    return _hybrid_predict_strips(df, with_evidence=True)


def predict_single(model_info: Dict, sample_df: pd.DataFrame) -> Dict:
    return batch_predict(model_info, sample_df)[0]


# ==================== 知识检索（含缓存） ====================
_FULLWIDTH_RE = re.compile(r"[Ａ-Ｚａ-ｚ０-９]")
_CN_RE = re.compile(r"[\u4e00-\u9fff]")


def _is_useful_chunk(text: str, min_chars: int = 20) -> bool:
    """过滤纯标点垃圾块和 PDF 全角英文提取噪音"""
    core = re.sub(r"[\W_]+", "", text or "")
    if len(core) < min_chars:
        return False
    fw = len(_FULLWIDTH_RE.findall(core))
    cn = len(_CN_RE.findall(core))
    return not (fw >= 10 and fw > cn)


def retrieve_context(retriever, query, feat_vec=None, fault_cn=None):
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
                    if not _is_useful_chunk(text):
                        continue
                    candidates.append({
                        'text': text,
                        'doc_type': h['entity'].get('doc_type', ''),
                        'score': h['distance'],
                    })
    if fault_cn:
        def_query = f"{fault_cn} 定义 特征 说明"
        def_vec = embed.encode(def_query, normalize_embeddings=True).tolist()
        def_docs = []
        for dt in ["document", "quadruple_text", "quadruple_kw"]:
            def_docs += client.search(
                collection_name="rag_knowledge", data=[def_vec],
                filter=f'doc_type == "{dt}"', limit=5, output_fields=["text"],
                search_params={"metric_type": "COSINE", "params": {"nprobe": 8}},
            )[0]
        exact_blocks = [h['entity']['text'] for h in def_docs if fault_cn in h['entity']['text']]
        if exact_blocks:
            forced = {
                'text': "\n\n".join(exact_blocks[:3]),
                'doc_type': 'exact_definition',
                'score': 2.0,
            }
            candidates = [c for c in candidates if c['text'] not in exact_blocks]
            candidates.insert(0, forced)

    def _pick_balanced(forced, normal, limit):
        """优先保留 exact 定义块，再保证每类 doc_type 至少 1 条，最后按重排分数补满。"""
        selected = list(forced)
        picked = {id(c) for c in selected}
        for dt in ["document", "quadruple_text", "quadruple_kw"]:
            for c in normal:
                if id(c) not in picked and c.get('doc_type') == dt:
                    selected.append(c)
                    picked.add(id(c))
                    break
        for c in normal:
            if len(selected) >= limit:
                break
            if id(c) not in picked:
                selected.append(c)
                picked.add(id(c))
        return selected[:limit]

    if candidates:
        forced_blocks = [c for c in candidates if c.get('doc_type') == 'exact_definition']
        normal_blocks = [c for c in candidates if c.get('doc_type') != 'exact_definition']
        if reranker and normal_blocks:
            normal_blocks = normal_blocks[:15]
            pairs = [(query if query else "", c['text']) for c in normal_blocks]
            scores = reranker.predict(pairs)
            for c, s in zip(normal_blocks, scores):
                c['rerank_score'] = float(s)
            normal_blocks.sort(key=lambda x: x.get('rerank_score', 0), reverse=True)
        candidates = _pick_balanced(forced_blocks, normal_blocks, TOP_K_RERANK)
    top_blocks = candidates
    context = "\n\n".join(
        f"[{i+1}] ({c['doc_type']}) {c['text']}" for i, c in enumerate(top_blocks)
    )
    MAX_CONTEXT_CHARS = 4000

    def truncate_context(ctx, max_chars):
        if len(ctx) <= max_chars:
            return ctx
        truncated = ctx[:max_chars]
        last_newline = truncated.rfind('\n')
        if last_newline > max_chars // 2:
            return truncated[:last_newline]
        return truncated

    context = truncate_context(context, MAX_CONTEXT_CHARS)

    @lru_cache(maxsize=128)
    def _search_similar(feat_vec_tuple):
        feat_list = list(feat_vec_tuple)
        sim_res = client.search(
            collection_name="fault_feature_vectors", data=[feat_list],
            limit=SIM_CASE_LIMIT, output_fields=["label", "plan"],
            search_params={"metric_type": "COSINE", "params": {"nprobe": 8}},
        )[0]
        if sim_res:
            sim_lines = [
                f"故障类型：{h['entity']['label']}，相似度：{h['distance']:.4f}，"
                f"方案：{h['entity']['plan']}"
                for h in sim_res
            ]
            return "【相似历史案例】\n" + "\n".join(sim_lines)
        return ""

    sim_text = ""
    if feat_vec and len(feat_vec) == 12:
        sim_text = _search_similar(tuple(feat_vec))
    return context, sim_text


# ==================== 检索结果缓存（不缓存空结果） ====================
_retrieve_cache: Dict[Tuple[str, str], Tuple[str, str]] = {}
_RETRIEVE_CACHE_TTL = 60 * 60 * 24  # 24 小时


def cached_retrieve_text(query: str, fault_cn: str = "") -> Tuple[str, str]:
    """带缓存的检索，但跳过空结果的缓存，避免空结果被永久命中。"""
    key = (query, fault_cn)
    cache_key = "rag:cache:retrieve:" + hashlib.sha1(
        json.dumps([query, fault_cn], ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    if MEMORY.enabled:
        cached = MEMORY.cache_get(cache_key)
        if cached and isinstance(cached, list) and len(cached) == 2:
            return tuple(cached)
    elif key in _retrieve_cache:
        return _retrieve_cache[key]

    context, sim_text = retrieve_context(
        _get_retriever(), query, feat_vec=None, fault_cn=fault_cn or None
    )
    if context:
        if MEMORY.enabled:
            MEMORY.cache_set(cache_key, list([context, sim_text]), ttl=_RETRIEVE_CACHE_TTL)
        _retrieve_cache[key] = (context, sim_text)
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
