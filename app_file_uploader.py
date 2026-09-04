#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
带钢宽度知识库在线更新（优化版）
- PDF：双栏友好解析 + 语义分块 + 1500 字符上限
- TXT：语义分块 + 1500 字符上限
- JSON 四元组：特征向量标准化后存入
"""

import streamlit as st
import fitz
import re
import json
import os
import tempfile
import numpy as np
import joblib
from typing import List, Dict
from pathlib import Path

from sentence_transformers import SentenceTransformer
from pymilvus import MilvusClient
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

try:
    from src.kb_utils import load_pdf_sorted_by_columns, build_quadruple_texts
except ImportError:
    from kb_utils import load_pdf_sorted_by_columns, build_quadruple_texts

# ==================== 配置 ====================
MILVUS_DB_PATH = str(Path(__file__).resolve().parent / "milvus_kb.db")
EMBED_MODEL_NAME = "models/bge-large-zh-v1.5"
DEVICE = "cuda" if __import__("torch").cuda.is_available() else "cpu"
MODEL_CONFIG_PATH = "./models/model_config.pkl"

CHUNK_SIZE = 800          # 保留显示用
CHUNK_OVERLAP = 100
MAX_CHUNK_CHARS = 1500    # 强制上限，与 build_kb.py 一致


def _invalidate_retrieve_cache() -> int:
    """知识库更新后使检索缓存失效（Redis / 文件 / 内存后端统一处理）。"""
    try:
        from src.memory_store import get_memory_store
        return get_memory_store().invalidate_retrieve_cache()
    except Exception:
        try:
            import sys as _sys
            from pathlib import Path as _Path
            _sys.path.insert(0, str(_Path(__file__).resolve().parent / "src"))
            from memory_store import get_memory_store
            return get_memory_store().invalidate_retrieve_cache()
        except Exception:
            return 0

# ==================== 页面基本设置 ====================
st.set_page_config(
    page_title="带钢宽度知识库更新",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ==================== 自定义工业风 CSS ====================
st.markdown("""
<style>
    .stApp { background-color: #F5F7FA; }
    .main-header {
        background: linear-gradient(90deg, #1E3A5F 0%, #2E5A88 100%);
        color: white; padding: 1.5rem 2rem; border-radius: 8px;
        margin-bottom: 2rem; display: flex; align-items: center; gap: 1rem;
    }
    .main-header h1 { color: white !important; font-size: 2rem; margin: 0; }
    .main-header p { color: #B0C4DE; margin: 0.2rem 0 0 0; }
    .upload-card {
        background: white; padding: 2rem; border-radius: 12px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.08); margin-bottom: 1.5rem;
    }
    .stButton > button {
        background: linear-gradient(90deg, #2E5A88 0%, #1E3A5F 100%);
        color: white; border: none; border-radius: 6px;
        padding: 0.5rem 2rem; font-weight: bold; transition: all 0.2s;
    }
    .stButton > button:hover {
        background: linear-gradient(90deg, #1E3A5F 0%, #2E5A88 100%);
        box-shadow: 0 4px 12px rgba(46,90,136,0.4);
    }
    .stTextArea textarea {
        background-color: #F0F4FA; font-family: 'Courier New', monospace;
    }
    .success-box {
        background: #E6F7E9; border-left: 4px solid #2E7D32;
        padding: 1rem; border-radius: 4px; margin: 1rem 0;
    }
    .info-box {
        background: #E3F0FF; border-left: 4px solid #1E3A5F;
        padding: 1rem; border-radius: 4px; margin: 1rem 0;
    }
    .upload-area {
        border: 2px dashed #B0C4DE; border-radius: 10px;
        padding: 2rem; text-align: center; background: #FAFBFD;
    }
</style>
""", unsafe_allow_html=True)

# ==================== 延迟加载资源 ====================
@st.cache_resource(show_spinner=False)
def get_embed_model():
    return SentenceTransformer(EMBED_MODEL_NAME, device=DEVICE)

@st.cache_resource(show_spinner=False)
def get_milvus_client():
    client = MilvusClient(MILVUS_DB_PATH)
    client.load_collection("rag_knowledge")
    client.load_collection("fault_feature_vectors")
    return client

@st.cache_resource(show_spinner=False)
def get_scaler():
    """加载训练时保存的标准化器，返回 (scaler, feature_names)"""
    if not os.path.exists(MODEL_CONFIG_PATH):
        return None, None
    config = joblib.load(MODEL_CONFIG_PATH)
    return config.get('scaler'), config.get('feature_names', [])

@st.cache_resource(show_spinner=False)
def get_chunker():
    """语义分块器，复用嵌入模型"""
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL_NAME,
        model_kwargs={'device': DEVICE},
        encode_kwargs={'normalize_embeddings': True}
    )
    return SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=70,
        buffer_size=3
    )

# ==================== 文档工具函数（优化版） ====================
def extract_text_ordered(pdf_bytes) -> str:
    """PDF 字节流 → 带版面解析的纯文本"""
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name
    try:
        docs = load_pdf_sorted_by_columns(tmp_path)
        return "\n\n".join([doc.page_content for doc in docs])
    finally:
        os.unlink(tmp_path)

def clean_text(text: str) -> str:
    text = re.sub(r'[ \u3000]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    lines = [line.strip() for line in text.split('\n')]
    return '\n'.join(lines)

def split_text(text: str) -> List[str]:
    """语义分块 + 最大长度强制切分"""
    chunker = get_chunker()
    doc = Document(page_content=text, metadata={"source": "uploaded"})
    chunks = chunker.split_documents([doc])

    fallback_splitter = RecursiveCharacterTextSplitter(
        chunk_size=MAX_CHUNK_CHARS,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", ".", " ", ""]
    )
    result = []
    for chunk in chunks:
        if len(chunk.page_content) <= MAX_CHUNK_CHARS:
            result.append(chunk.page_content)
        else:
            sub_texts = fallback_splitter.split_text(chunk.page_content)
            result.extend(sub_texts)
    return result

def insert_chunks_to_milvus(chunks: List[str], metadata: dict = None):
    if not chunks:
        return
    embed_model = get_embed_model()
    client = get_milvus_client()

    progress_bar = st.progress(0)
    status_text = st.empty()
    embeddings = []
    batch_size = 32
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        batch_emb = embed_model.encode(batch, normalize_embeddings=True)
        embeddings.extend(batch_emb)
        progress_bar.progress(min((i + batch_size) / len(chunks), 1.0))
        status_text.text(f"向量化中... {min(i+batch_size, len(chunks))}/{len(chunks)}")

    data = []
    for text, emb in zip(chunks, embeddings):
        data.append({
            "text": text,
            "embedding": emb.tolist(),
            "doc_type": "document",
            "metadata": metadata or {"source": "uploaded"}
        })

    status_text.text("正在插入数据库...")
    client.insert("rag_knowledge", data)
    try:
        client.flush("rag_knowledge")
    except Exception:
        pass
    progress_bar.empty()
    status_text.empty()
    n = _invalidate_retrieve_cache()
    if n:
        st.caption(f"已失效 {n} 条检索缓存，知识库更新立即生效。")

# ==================== 四元组处理（优化版） ====================
def insert_quadruples(quads: List[Dict]):
    if not quads:
        return
    embed_model = get_embed_model()
    client = get_milvus_client()

    desc_texts, kw_texts = [], []
    feat_vectors, feat_labels, feat_plans = [], [], []
    scaler, feature_names = get_scaler()
    for q in quads:
        processed = build_quadruple_texts(q, scaler, feature_names)
        desc_texts.append(processed["description"])
        kw_texts.append(processed["keywords"])
        if processed["feature_vector_scaled"]:
            feat_vectors.append(processed["feature_vector_scaled"])
            feat_labels.append(processed["故障类型"])
            feat_plans.append(processed["诊断方案"])

    status_text = st.empty()
    status_text.text("正在向量化四元组...")
    desc_embeddings = embed_model.encode(desc_texts, normalize_embeddings=True).tolist()
    kw_embeddings = embed_model.encode(kw_texts, normalize_embeddings=True).tolist()

    desc_data, kw_data = [], []
    for i in range(len(quads)):
        meta = {
            "设备": quads[i].get("设备", ""),
            "故障类型": quads[i].get("故障类型", ""),
            "故障特征": quads[i].get("故障特征", "").replace("，", ",")
        }
        desc_data.append({"text": desc_texts[i], "embedding": desc_embeddings[i], "doc_type": "quadruple_text", "metadata": meta})
        kw_data.append({"text": kw_texts[i], "embedding": kw_embeddings[i], "doc_type": "quadruple_kw", "metadata": meta})

    client.insert("rag_knowledge", desc_data)
    client.insert("rag_knowledge", kw_data)

    if feat_vectors:
        feat_insert = []
        for vec, label, plan in zip(feat_vectors, feat_labels, feat_plans):
            feat_insert.append({"vector": vec, "label": label, "plan": plan})
        client.insert("fault_feature_vectors", feat_insert)

    try:
        client.flush("rag_knowledge")
        client.flush("fault_feature_vectors")
    except Exception:
        pass

    status_text.empty()
    st.markdown(f'<div class="success-box">✅ 四元组更新完成：{len(quads)} 条描述/关键词已添加，{len(feat_vectors)} 个特征向量已更新。</div>', unsafe_allow_html=True)
    n = _invalidate_retrieve_cache()
    if n:
        st.caption(f"已失效 {n} 条检索缓存，知识库更新立即生效。")

# ==================== 页面主体 ====================
st.markdown(
    """
    <div class="main-header">
        <span style="font-size: 2.5rem;">🏭</span>
        <div>
            <h1>带钢宽度知识库管理系统</h1>
            <p>在线更新技术文档与故障诊断方案</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown('<div class="upload-card">', unsafe_allow_html=True)
    st.subheader("📁 文件上传")
    uploader_file = st.file_uploader(
        "支持 PDF / TXT 文档或 JSON 四元组文件",
        type=['txt', 'pdf', 'json'],
        accept_multiple_files=False,
        label_visibility="collapsed"
    )
    if uploader_file is not None:
        file_name = uploader_file.name
        ext = file_name.split('.')[-1].lower()
        file_size = uploader_file.size / 1024
        st.markdown(f"**已选择：** `{file_name}` （{ext.upper()}，{file_size:.2f} KB）")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="upload-card">', unsafe_allow_html=True)
    st.subheader("ℹ️ 使用说明")
    st.markdown("""
    - **技术文档**：PDF/TXT，自动分块并向量化存入知识库。
    - **四元组文件**：JSON 格式，包含设备、故障类型、特征和诊断方案，实时更新故障知识库。
    - 上传后点击下方按钮完成知识注入。
    """)
    st.markdown('</div>', unsafe_allow_html=True)

if uploader_file is not None:
    file_name = uploader_file.name
    ext = file_name.split('.')[-1].lower()

    try:
        if ext == 'json':
            raw_data = json.load(uploader_file)
            if isinstance(raw_data, list):
                quads = raw_data
            elif isinstance(raw_data, dict) and 'data' in raw_data:
                quads = raw_data['data']
            else:
                st.error("JSON 格式错误：应为四元组数组。")
                st.stop()

            st.info(f"已加载 {len(quads)} 条四元组记录")
            if st.button("📥 确认更新四元组知识库", type="primary"):
                with st.spinner("正在处理四元组数据..."):
                    insert_quadruples(quads)

        elif ext in ['txt', 'pdf']:
            if ext == 'txt':
                raw = uploader_file.getvalue().decode("utf-8")
                content = clean_text(raw)
            elif ext == 'pdf':
                raw = extract_text_ordered(uploader_file.getvalue())
                if not raw.strip():
                    st.warning("未能提取文本，可能是扫描件，建议使用 OCR 处理。")
                    content = ""
                else:
                    content = clean_text(raw)

            if content:
                st.markdown("### 文档预览")
                st.text_area("内容（前 5000 字符）", content[:5000], height=300, disabled=True)
                if st.button("📥 确认更新文档知识库", type="primary"):
                    with st.spinner("正在分块并向量化..."):
                        chunks = split_text(content)
                        st.info(f"文档被分为 {len(chunks)} 个块")
                    insert_chunks_to_milvus(chunks, metadata={"source": file_name})
                    st.markdown(f'<div class="success-box">✅ 文档知识库更新成功！已添加 {len(chunks)} 个新片段。</div>', unsafe_allow_html=True)
            else:
                st.info("文件内容为空，无法更新。")
        else:
            st.error("不支持的文件格式")
    except Exception as e:
        st.error(f"处理失败：{e}")
else:
    st.markdown(
        """
        <div class="info-box">
            <strong>💡 提示：</strong>请上传一个 PDF、TXT 或 JSON 文件，系统将自动识别类型并完成知识入库。
            上传后可在“交互诊断系统”中立即生效。
        </div>
        """,
        unsafe_allow_html=True
    )
