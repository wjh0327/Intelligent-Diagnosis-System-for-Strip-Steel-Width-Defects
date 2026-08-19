#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
带钢宽度缺陷诊断 RAG 知识库构建脚本（标准化版）
====================================
1. 解析 documents/ 下所有 PDF (双栏友好) / DOCX / TXT / MD 文档，语义分块并存入 Milvus
2. 读取 quadruples.json，将四元组转为多形态文本块并存入 Milvus
3. 将故障特征数值向量标准化后存入另一个 Collection 供相似案例检索

修复：
- 建议2：PDF 双栏排序、单栏兼容、页眉页脚过滤、全宽块处理
- 建议1：语义分块后强制最大长度，防止超大块撑爆上下文窗口
- 嵌入模型复用：只加载一次 HuggingFaceEmbeddings，消除内存浪费
"""

import os
import json
import re
import warnings
import numpy as np
import joblib
from pathlib import Path
from typing import List, Dict, Any

# 文档解析
from langchain_community.document_loaders import (
    TextLoader,
    UnstructuredMarkdownLoader,
    Docx2txtLoader,          # 新增 DOCX 支持
)
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

# PDF 解析（双栏友好）
import fitz  # PyMuPDF

# Milvus
from pymilvus import MilvusClient, DataType
import logging

logging.getLogger("pypdf").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

# ==================== 配置参数 ====================
DOCUMENTS_DIR = "./data/documents"               # 文档目录
QUADRUPLE_JSON = "./data/quadruplets/width.json" # 四元组文件
MILVUS_DB_PATH = r"F:\RAG Agent\milvus_kb.db"               # 数据库文件

EMBED_MODEL_NAME = "models/bge-large-zh-v1.5"
EMBED_DEVICE = "cuda" if __import__("torch").cuda.is_available() else "cpu"

MODEL_CONFIG_PATH = "./models/model_config.pkl"  # 训练好的模型配置（含 scaler）

# 建议1：单块最大字符数，防止上下文窗口溢出
MAX_CHUNK_CHARS = 1500
CHUNK_OVERLAP = 100

COLLECTION_KNOWLEDGE = "rag_knowledge"
COLLECTION_FEATURES = "fault_feature_vectors"

# 过滤纯标点垃圾块和 PDF 全角英文提取噪音
_FULLWIDTH_RE = re.compile(r"[Ａ-Ｚａ-ｚ０-９]")
_CN_RE = re.compile(r"[\u4e00-\u9fff]")

def _is_useful_text(text: str, min_chars: int = 20) -> bool:
    core = re.sub(r"[\W_]+", "", text or "")
    if len(core) < min_chars:
        return False
    fw = len(_FULLWIDTH_RE.findall(core))
    cn = len(_CN_RE.findall(core))
    return not (fw >= 10 and fw > cn)


# ==================== 工具函数 ====================
def load_pdf_sorted_by_columns(file_path: str) -> List[Document]:
    """
    加载 PDF 并按双栏逻辑排序，同时处理单栏文档、跨栏标题、页眉页脚。
    每页生成一个 Document，保留页码元数据。
    """
    docs = []
    pdf_doc = fitz.open(file_path)
    file_name = Path(file_path).name

    for page_num in range(len(pdf_doc)):
        page = pdf_doc[page_num]
        width = page.rect.width
        height = page.rect.height
        mid_x = width / 2

        # 提取所有文本块
        blocks = page.get_text("blocks")
        text_blocks = [b for b in blocks if b[6] == 0]  # 仅文本块

        # 过滤页眉/页脚（高度阈值可调节）
        filtered_blocks = []
        for b in text_blocks:
            _, y0, _, y1, _, _, _ = b
            if y0 < height * 0.1 or y1 > height * 0.9:
                continue  # 丢弃页眉/页脚
            filtered_blocks.append(b)
        text_blocks = filtered_blocks

        if not text_blocks:
            continue

        # 识别全宽块（跨栏标题、宽图注等）
        # 宽松条件：宽度 > 60% 页宽，且中心点接近页面中线 (偏离中线 < 20% 页宽)
        full_width_blocks = []
        normal_blocks = []
        for b in text_blocks:
            x0, y0, x1, y1, text, _, _ = b
            block_width = x1 - x0
            block_center = (x0 + x1) / 2
            if block_width > width * 0.6 and abs(block_center - mid_x) < width * 0.2:
                full_width_blocks.append((y0, text))
            else:
                normal_blocks.append(b)

        # 检测是否为双栏：左右半区都有块，且全宽块不太多
        left_blocks_raw = [b for b in normal_blocks if b[0] < mid_x]
        right_blocks_raw = [b for b in normal_blocks if b[0] >= mid_x]
        is_two_column = (
            len(left_blocks_raw) > 0 and
            len(right_blocks_raw) > 0 and
            len(full_width_blocks) < 6   # 若全宽块过多，视为单栏
        )

        # 正文排序
        if is_two_column:
            left_blocks = sorted(left_blocks_raw, key=lambda b: b[1])
            right_blocks = sorted(right_blocks_raw, key=lambda b: b[1])
            body_text = (
                "\n".join(b[4] for b in left_blocks) +
                "\n" +
                "\n".join(b[4] for b in right_blocks)
            )
        else:
            # 单栏直接按 y 排序
            sorted_blocks = sorted(normal_blocks, key=lambda b: b[1])
            body_text = "\n".join(b[4] for b in sorted_blocks)

        # 处理全宽块：按 y 坐标分为顶部块和底部块，分别放于正文前后
        if full_width_blocks:
            full_width_blocks.sort(key=lambda x: x[0])   # 按 y 坐标排序
            # 以页面高度的 50% 为界，分为顶部标题和底部注释
            top_blocks = [t for y, t in full_width_blocks if y < height * 0.5]
            bottom_blocks = [t for y, t in full_width_blocks if y >= height * 0.5]
            parts = []
            if top_blocks:
                parts.append("\n".join(top_blocks))
            parts.append(body_text)
            if bottom_blocks:
                parts.append("\n".join(bottom_blocks))
            page_text = "\n\n".join(parts)
        else:
            page_text = body_text

        if page_text.strip():
            docs.append(Document(
                page_content=page_text,
                metadata={
                    "source": file_name,
                    "page": page_num + 1
                }
            ))

    pdf_doc.close()
    return docs


def load_documents(dir_path: str) -> List[Document]:
    """加载目录下所有支持格式的文件：PDF(双栏友好)、DOCX、TXT、MD"""
    all_docs = []
    dir_path = Path(dir_path)

    # --- PDF 文件（双栏排序）---
    pdf_files = list(dir_path.glob("**/*.pdf"))
    for pdf_file in pdf_files:
        try:
            docs = load_pdf_sorted_by_columns(str(pdf_file))
            all_docs.extend(docs)
        except Exception as e:
            print(f"  加载 PDF 失败 {pdf_file}: {e}")
    print(f"  加载 PDF 文件 {len(pdf_files)} 个")

    # --- DOCX 文件 ---
    docx_files = list(dir_path.glob("**/*.docx"))
    for docx_file in docx_files:
        try:
            loader = Docx2txtLoader(str(docx_file))
            docs = loader.load()
            for doc in docs:
                doc.metadata["source_file"] = str(docx_file.name)
            all_docs.extend(docs)
        except Exception as e:
            print(f"  加载 DOCX 失败 {docx_file}: {e}")
    print(f"  加载 DOCX 文件 {len(docx_files)} 个")

    # --- TXT 文件 ---
    txt_files = list(dir_path.glob("**/*.txt"))
    for txt_file in txt_files:
        try:
            loader = TextLoader(str(txt_file), autodetect_encoding=True)
            docs = loader.load()
            for doc in docs:
                doc.metadata["source_file"] = str(txt_file.name)
            all_docs.extend(docs)
        except Exception as e:
            print(f"  加载 TXT 失败 {txt_file}: {e}")
    print(f"  加载 TXT 文件 {len(txt_files)} 个")

    # --- Markdown 文件 ---
    md_files = list(dir_path.glob("**/*.md"))
    for md_file in md_files:
        try:
            loader = UnstructuredMarkdownLoader(str(md_file))
            docs = loader.load()
            for doc in docs:
                doc.metadata["source_file"] = str(md_file.name)
            all_docs.extend(docs)
        except Exception as e:
            print(f"  加载 MD 失败 {md_file}: {e}")
    print(f"  加载 MD 文件 {len(md_files)} 个")

    print(f"共加载原始文档 {len(all_docs)} 个")
    return all_docs


def load_quadruples(json_path: str) -> List[Dict]:
    """加载四元组 JSON，返回列表"""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"加载四元组 {len(data)} 条")
    return data


# 四元组 JSON / 原始 CSV 的列序（与 test_samples.csv 表头一致）
QUAD_FEATURE_ORDER = [
    "is_FM", "is_RM", "is_DM", "is_HT", "is_WS", "is_NA",
    "FMWTARGETHOT", "RDWTARGETTOTAL", "PDIWIDTHTOL",
    "FMWIDTHACTHOT", "RMWIDTHACTHOT", "FMWTARGETCOL",
]


def reorder_features_to_model_order(raw_vector: List[float],
                                    feature_names: List[str]) -> List[float]:
    """
    将四元组/CSV 列序的 12 维特征重排为模型 feature_names 列序。
    必须与 scaler 使用同一坐标系，否则相似案例检索的余弦距离无意义。
    """
    if len(raw_vector) != 12 or len(feature_names) != 12:
        return raw_vector
    try:
        pos = [QUAD_FEATURE_ORDER.index(name) for name in feature_names]
        return [raw_vector[i] for i in pos]
    except ValueError:
        return raw_vector


def build_quadruple_texts(quad: Dict, scaler, feature_names: List[str] = None) -> Dict[str, Any]:
    """
    将单条四元组转为文本和标准化特征向量
    """
    device = quad.get("设备", "")
    fault_type = quad.get("故障类型", "")
    feature_str = quad.get("故障特征", "")
    solution = quad.get("诊断方案", "")

    description = (
        f"设备：{device}。故障类型：{fault_type}。"
        f"故障特征向量：{feature_str}。诊断方案：{solution}"
    )
    keywords = f"设备:{device} 故障类型:{fault_type} 特征:{feature_str}"

    # 解析原始特征向量
    raw_vector = []
    try:
        raw_vector = [float(x.strip()) for x in feature_str.replace("，", ",").split(",") if x.strip()]
    except ValueError:
        print(f"警告: 四元组故障特征解析失败: {feature_str}")

    # 先重排为模型 feature_names 列序，再标准化（与查询端坐标系一致）
    if raw_vector and feature_names:
        raw_vector = reorder_features_to_model_order(raw_vector, feature_names)

    # 标准化特征向量（若 scaler 存在且向量有效）
    scaled_vector = []
    if raw_vector and scaler is not None:
        try:
            scaled_vector = scaler.transform([raw_vector])[0].tolist()
        except Exception as e:
            print(f"标准化失败: {e}, 使用原始值")
            scaled_vector = raw_vector
    else:
        scaled_vector = raw_vector

    return {
        "description": description,
        "keywords": keywords,
        "feature_vector_raw": raw_vector,
        "feature_vector_scaled": scaled_vector,
        "设备": device,
        "故障类型": fault_type,
        "故障特征": feature_str,
        "诊断方案": solution
    }


def batch_embed(embeddings: HuggingFaceEmbeddings, texts: List[str],
                batch_size: int = 128) -> List[List[float]]:
    """批量嵌入，返回归一化后的向量列表"""
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        # embed_documents 返回的是 numpy 数组或列表，确保为 Python 原生列表
        vecs = embeddings.embed_documents(batch)
        all_embeddings.extend([v.tolist() if isinstance(v, np.ndarray) else v for v in vecs])
    return all_embeddings


# ==================== 主流程 ====================
def main():
    # 1. 加载模型配置（主要获取 scaler）
    print("加载模型配置...")
    if not os.path.exists(MODEL_CONFIG_PATH):
        raise FileNotFoundError(f"模型配置文件不存在: {MODEL_CONFIG_PATH}")
    model_config = joblib.load(MODEL_CONFIG_PATH)
    scaler = model_config.get('scaler')
    feature_names = model_config.get('feature_names', [])
    print(f"Scaler 已加载，特征数量: {len(feature_names)}")

    # 2. 初始化嵌入模型（只加载一次）
    print(f"正在加载嵌入模型 {EMBED_MODEL_NAME} ...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL_NAME,
        model_kwargs={'device': EMBED_DEVICE},
        encode_kwargs={'normalize_embeddings': True}  # 归一化输出
    )
    print(f"嵌入模型加载完成，输出维度: 1024")  # bge-large-zh 为 1024

    # 3. 连接 Milvus Lite
    print(f"连接 Milvus Lite: {MILVUS_DB_PATH}")
    client = MilvusClient(MILVUS_DB_PATH)

    # 4. 加载文档并语义分块
    print("\n===== 1/4 处理文档 =====")
    raw_docs = load_documents(DOCUMENTS_DIR)

    print("正在进行语义分块...")
    # 直接复用上面加载的 embeddings，不再重复加载
    chunker = SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=70,
        buffer_size=3   # 增大缓冲区以获得更好的语义连贯性
    )
    doc_chunks = chunker.split_documents(raw_docs)
    print(f"文档分块完成，共 {len(doc_chunks)} 个块")

    # 建议1：强制限制分块最大长度，防止超大块
    fallback_splitter = RecursiveCharacterTextSplitter(
        chunk_size=MAX_CHUNK_CHARS,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", ".", " ", ""]
    )
    safe_chunks = []
    skipped_junk = 0
    for chunk in doc_chunks:
        parts = [chunk] if len(chunk.page_content) <= MAX_CHUNK_CHARS else [
            Document(page_content=sub, metadata=chunk.metadata)
            for sub in fallback_splitter.split_text(chunk.page_content)
        ]
        for part in parts:
            if not _is_useful_text(part.page_content):
                skipped_junk += 1
                continue  # 跳过纯标点/空白的垃圾分块
            safe_chunks.append(part)
    print(f"分块安全检查后，块数从 {len(doc_chunks)} → {len(safe_chunks)}（跳过垃圾块 {skipped_junk} 个）")
    doc_chunks = safe_chunks

    # 5. 加载并处理四元组（标准化特征向量）
    print("\n===== 2/4 处理四元组 =====")
    quadruples_raw = load_quadruples(QUADRUPLE_JSON)
    quad_processed = [build_quadruple_texts(q, scaler, feature_names) for q in quadruples_raw]

    # 提取各类文本
    doc_texts = [chunk.page_content for chunk in doc_chunks]
    doc_metas = [chunk.metadata for chunk in doc_chunks]

    quad_desc_texts = [q["description"] for q in quad_processed]
    quad_kw_texts = [q["keywords"] for q in quad_processed]
    quad_metas = [{k: q[k] for k in ("设备", "故障类型", "故障特征")}
                  for q in quad_processed]

    # 提取标准化后的特征向量（用于相似案例检索）
    valid_feat_quad = [q for q in quad_processed if len(q["feature_vector_scaled"]) > 0]
    feat_vectors_scaled = [q["feature_vector_scaled"] for q in valid_feat_quad]
    feat_labels = [q["故障类型"] for q in valid_feat_quad]
    feat_plans = [q["诊断方案"] for q in valid_feat_quad]

    if feat_vectors_scaled:
        feat_dim = len(feat_vectors_scaled[0])
        print(f"有效故障特征向量 {len(feat_vectors_scaled)} 条，维度 {feat_dim}")
    else:
        feat_dim = 0

    # 6. 批量嵌入（复用同一个 embeddings 实例）
    print("\n===== 3/4 向量化所有文本 =====")
    print("嵌入文档块...")
    doc_embeds = batch_embed(embeddings, doc_texts)
    print("嵌入四元组描述...")
    quad_desc_embeds = batch_embed(embeddings, quad_desc_texts)
    print("嵌入四元组关键词...")
    quad_kw_embeds = batch_embed(embeddings, quad_kw_texts)

    # 7. 创建 Milvus 集合并插入数据
    print("\n===== 4/4 存入 Milvus =====")

    # 7.1 创建主知识库集合
    if client.has_collection(COLLECTION_KNOWLEDGE):
        client.drop_collection(COLLECTION_KNOWLEDGE)
    schema_know = client.create_schema(auto_id=True)
    schema_know.add_field("id", DataType.INT64, is_primary=True)
    schema_know.add_field("text", DataType.VARCHAR, max_length=8000)
    schema_know.add_field("embedding", DataType.FLOAT_VECTOR, dim=1024)
    schema_know.add_field("doc_type", DataType.VARCHAR, max_length=50)
    schema_know.add_field("metadata", DataType.JSON)

    index_know = client.prepare_index_params()
    index_know.add_index("embedding", index_type="FLAT",
                         metric_type="COSINE")
    client.create_collection(COLLECTION_KNOWLEDGE, schema=schema_know,
                             index_params=index_know)
    print(f"集合 {COLLECTION_KNOWLEDGE} 创建成功")

    # 插入文档块
    doc_data = [{"text": t, "embedding": e, "doc_type": "document", "metadata": m}
                for t, m, e in zip(doc_texts, doc_metas, doc_embeds)]
    if doc_data:
        client.insert(COLLECTION_KNOWLEDGE, doc_data)
        print(f"已插入 {len(doc_data)} 个文档块")

    # 插入四元组描述
    quad_desc_data = [{"text": t, "embedding": e, "doc_type": "quadruple_text", "metadata": m}
                      for t, m, e in zip(quad_desc_texts, quad_metas, quad_desc_embeds)]
    if quad_desc_data:
        client.insert(COLLECTION_KNOWLEDGE, quad_desc_data)
        print(f"已插入 {len(quad_desc_data)} 条四元组描述")

    # 插入四元组关键词
    quad_kw_data = [{"text": t, "embedding": e, "doc_type": "quadruple_kw", "metadata": m}
                    for t, m, e in zip(quad_kw_texts, quad_metas, quad_kw_embeds)]
    if quad_kw_data:
        client.insert(COLLECTION_KNOWLEDGE, quad_kw_data)
        print(f"已插入 {len(quad_kw_data)} 条四元组关键词")

    # 7.2 创建故障特征向量集合（存储标准化后的向量）
    if feat_vectors_scaled and feat_dim > 0:
        if client.has_collection(COLLECTION_FEATURES):
            client.drop_collection(COLLECTION_FEATURES)

        schema_feat = client.create_schema(auto_id=True)
        schema_feat.add_field("id", DataType.INT64, is_primary=True)
        schema_feat.add_field("vector", DataType.FLOAT_VECTOR, dim=feat_dim)
        schema_feat.add_field("label", DataType.VARCHAR, max_length=100)
        schema_feat.add_field("plan", DataType.VARCHAR, max_length=4000)

        index_feat = client.prepare_index_params()
        index_feat.add_index("vector", index_type="IVF_FLAT",
                             metric_type="COSINE", params={"nlist": 64})
        client.create_collection(COLLECTION_FEATURES, schema=schema_feat,
                                 index_params=index_feat)
        print(f"集合 {COLLECTION_FEATURES} 创建成功，维度 {feat_dim}")

        feat_insert_data = [
            {"vector": vec, "label": label, "plan": plan}
            for vec, label, plan in zip(feat_vectors_scaled, feat_labels, feat_plans)
        ]
        client.insert(COLLECTION_FEATURES, feat_insert_data)
        print(f"已插入 {len(feat_insert_data)} 条标准化故障特征向量")
    else:
        print("无有效特征向量，跳过特征集合创建")

    # 8. 持久化并关闭
    client.close()
    print("\n知识库构建完成！")


if __name__ == "__main__":
    main()
