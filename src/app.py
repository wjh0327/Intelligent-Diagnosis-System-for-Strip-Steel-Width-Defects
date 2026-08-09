#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
带钢宽度缺陷诊断系统 - 自包含完整版
包含模型加载、检索、Agent 图构建，支持批量诊断结果持久化。
"""

import os
import json
import warnings
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import torch
from typing import List, Dict, Any, Generator, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache

import streamlit as st
from model_def import BalancedTraceabilityNN, ModelEnsemble
from sentence_transformers import SentenceTransformer, CrossEncoder
from pymilvus import MilvusClient
import dashscope
from dashscope import Generation

from tools import create_tools
from agent_graph import build_agent_graph

warnings.filterwarnings("ignore")

# ==================== 全局路径配置 ====================
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
MODEL_DIR = str(PROJECT_ROOT / "models")
MILVUS_DB = str(PROJECT_ROOT / "milvus_kb.db")
EMBED_MODEL_PATH = str(PROJECT_ROOT / "models" / "bge-large-zh-v1.5")
RERANKER_MODEL_PATH = str(PROJECT_ROOT / "models" / "bge-reranker-v2-m3")
SKILLS_DIR = str(PROJECT_ROOT / "skills")

LLM_MODEL = "qwen3.7-max"  # 可改为 "deepseek-v4-pro" 等
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")

TOP_K_RETRIEVAL = 8
TOP_K_RERANK = 4
SIM_CASE_LIMIT = 5
MAX_LLM_WORKERS = 5

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

# ==================== 资源加载与缓存 ====================
@st.cache_resource(show_spinner=False)
def load_classification_model() -> Dict[str, Any]:
    config_path = os.path.join(MODEL_DIR, "model_config.pkl")
    if not os.path.exists(config_path):
        st.error(f"模型配置文件不存在: {config_path}")
        st.stop()
    config = joblib.load(config_path)
    feature_names = config['feature_names']
    n_features = len(feature_names)
    n_classes = config['n_classes']
    label_encoder = config['label_encoder']
    scaler = config['scaler']
    label_mapping = config.get('label_mapping', {})
    problem_descriptions = config.get('problem_descriptions', {})
    mic_weights = config.get('mic_weights', None)
    class_weights = config.get('class_weights', None)
    ensemble_info = config.get('ensemble_info', {'enabled': False, 'size': 1, 'method': 'soft_voting'})
    use_ensemble = ensemble_info.get('enabled', False) and ensemble_info.get('size', 1) > 1

    model_params = config.get('model_params', {})
    if not model_params:
        model_params = {
            'n_features': n_features,
            'n_classes': n_classes,
            'mic_weights': mic_weights,
            'hidden_dims': [256, 128, 64],
            'dropout_rate': 0.35,
            'use_attention': True,
            'attention_learnable': True,
            'use_focal_loss': config.get('training_config', {}).get('use_focal_loss', False),
            'gamma': config.get('training_config', {}).get('focal_gamma', 2.5),
            'alpha': class_weights,
        }
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if use_ensemble:
        models = []
        for i in range(ensemble_info['size']):
            weight_path = os.path.join(MODEL_DIR, f"ensemble_best_model_{i+1}.pth")
            m = BalancedTraceabilityNN(**model_params)
            m.load_state_dict(torch.load(weight_path, map_location='cpu'))
            m.eval().to(device)
            models.append(m)
        model = ModelEnsemble(models, None, ensemble_method=ensemble_info['method'])
    else:
        weight_path = os.path.join(MODEL_DIR, "best_traceability_model.pth")
        if not os.path.exists(weight_path):
            st.error(f"模型权重文件不存在: {weight_path}")
            st.stop()
        model = BalancedTraceabilityNN(**model_params)
        model.load_state_dict(torch.load(weight_path, map_location='cpu'))
        model.eval().to(device)

    importance_cache = {}
    try:
        if use_ensemble:
            attn_weights = model.models[0].attention.get_attention_matrix()
        else:
            attn_weights = model.attention.get_attention_matrix()
        for idx, fault_cn in label_mapping.items():
            imp = [(str(feature_names[i]), float(attn_weights[idx][i])) for i in range(n_features)]
            imp.sort(key=lambda x: x[1], reverse=True)
            importance_cache[fault_cn] = imp
    except Exception:
        pass

    return {
        'model': model,
        'feature_names': feature_names,
        'label_encoder': label_encoder,
        'scaler': scaler,
        'label_mapping': label_mapping,
        'problem_descriptions': problem_descriptions,
        'use_ensemble': use_ensemble,
        'device': device,
        'importance_cache': importance_cache
    }

@st.cache_resource(show_spinner=False)
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
    model = model_info['model']
    feature_names = model_info['feature_names']
    scaler = model_info['scaler']
    label_mapping = model_info['label_mapping']
    problem_descriptions = model_info['problem_descriptions']
    use_ensemble = model_info['use_ensemble']
    device = model_info['device']

    X = df[feature_names].values.astype(np.float32)
    X_scaled = scaler.transform(X)
    results = []
    for i in range(0, len(X_scaled), batch_size):
        batch = X_scaled[i:i+batch_size]
        X_tensor = torch.from_numpy(batch).to(device)
        if use_ensemble:
            preds = model.predict(batch, device)
            pred_classes = preds['predictions']
            probs_batch = preds['probabilities']
        else:
            with torch.no_grad():
                out = model.predict_with_explanation(X_tensor, device)
                pred_classes = out['predictions']
                probs_batch = out['probabilities']
        for j in range(len(batch)):
            pred = int(pred_classes[j])
            fault_cn = label_mapping.get(pred, str(pred))
            fault_desc = problem_descriptions.get(fault_cn, "")
            results.append({
                'pred_class': pred,
                'fault_cn': fault_cn,
                'fault_desc': fault_desc,
                'probs': probs_batch[j].tolist() if probs_batch is not None else None,
                'feat_vector': batch[j].tolist()
            })
    return results

def predict_single(model_info: Dict, sample_df: pd.DataFrame) -> Dict:
    return batch_predict(model_info, sample_df)[0]

# ==================== 知识检索（含缓存） ====================
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
                search_params={"metric_type": "COSINE", "params": {"nprobe": 8}}
            )[0]
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {executor.submit(search_doc_type, dt): dt for dt in doc_types}
            for future in as_completed(futures):
                for h in future.result():
                    candidates.append({
                        'text': h['entity']['text'],
                        'doc_type': h['entity'].get('doc_type', ''),
                        'score': h['distance']
                    })
    if fault_cn:
        def_query = f"{fault_cn} 定义 特征 说明"
        def_vec = embed.encode(def_query, normalize_embeddings=True).tolist()
        def_docs = []
        for dt in ["document", "quadruple_text", "quadruple_kw"]:
            def_docs += client.search(
                collection_name="rag_knowledge", data=[def_vec],
                filter=f'doc_type == "{dt}"', limit=5, output_fields=["text"],
                search_params={"metric_type": "COSINE", "params": {"nprobe": 8}}
            )[0]
        exact_blocks = [h['entity']['text'] for h in def_docs if fault_cn in h['entity']['text']]
        if exact_blocks:
            forced = {
                'text': "\n\n".join(exact_blocks[:3]),
                'doc_type': 'exact_definition',
                'score': 2.0
            }
            candidates = [c for c in candidates if c['text'] not in exact_blocks]
            candidates.insert(0, forced)
    if reranker and candidates:
        forced_blocks = [c for c in candidates if c.get('doc_type') == 'exact_definition']
        normal_blocks = [c for c in candidates if c.get('doc_type') != 'exact_definition']
        if normal_blocks:
            normal_blocks = normal_blocks[:20]
            pairs = [(query if query else "", c['text']) for c in normal_blocks]
            scores = reranker.predict(pairs)
            for c, s in zip(normal_blocks, scores):
                c['rerank_score'] = float(s)
            normal_blocks.sort(key=lambda x: x.get('rerank_score', 0), reverse=True)
        candidates = forced_blocks + normal_blocks
    top_blocks = candidates[:TOP_K_RERANK]
    context = "\n\n".join(f"[{i+1}] ({c['doc_type']}) {c['text']}" for i, c in enumerate(top_blocks))
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
            search_params={"metric_type": "COSINE", "params": {"nprobe": 8}}
        )[0]
        if sim_res:
            sim_lines = [
                f"故障类型：{h['entity']['label']}，相似度：{h['distance']:.4f}，方案：{h['entity']['plan']}"
                for h in sim_res
            ]
            return "【相似历史案例】\n" + "\n".join(sim_lines)
        return ""
    sim_text = ""
    if feat_vec and len(feat_vec) == 12:
        sim_text = _search_similar(tuple(feat_vec))
    return context, sim_text

@lru_cache(maxsize=128)
def cached_retrieve_text(query: str, fault_cn: str = ""):
    fc = fault_cn if fault_cn else None
    return retrieve_context(retriever, query, feat_vec=None, fault_cn=fc)

# ==================== 初始化全局资源（Agent 图） ====================
@st.cache_resource(show_spinner="正在加载模型与知识库...")
def init_resources():
    model_info = load_classification_model()
    retriever = load_retriever()
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
        api_key=DASHSCOPE_API_KEY,
        max_steps=6,
        skills_dir=SKILLS_DIR
    )
    return model_info, retriever, feature_names, agent_graph

model_info, retriever, feature_names, agent_graph = init_resources()

# ==================== 会话状态初始化 ====================
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'csv_path' not in st.session_state:
    st.session_state.csv_path = None
if 'batch_done' not in st.session_state:
    st.session_state.batch_done = False
if 'batch_summary' not in st.session_state:
    st.session_state.batch_summary = ""

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

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

uploaded_file = st.file_uploader("📎 上传CSV文件（可选）", type="csv", key="csv_uploader")
if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        tmp.write(uploaded_file.getvalue())
        st.session_state.csv_path = tmp.name
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
    st.rerun()

if st.button("🗑️ 清除对话历史"):
    st.session_state.messages = []
    st.session_state.csv_path = None
    st.session_state.batch_done = False
    st.session_state.batch_summary = ""
    st.rerun()