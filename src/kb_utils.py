# -*- coding: utf-8 -*-
"""
src/kb_utils.py —— 知识库公共工具
==================================
从 build_kb.py / app_file_uploader.py 抽取的共享实现：
  - load_pdf_sorted_by_columns：双栏 PDF 解析
  - reorder_features_to_model_order：四元组/CSV 列序 → 模型列序
  - build_quadruple_texts：四元组 → 文本块 + 标准化特征向量
"""

import logging
from pathlib import Path
from typing import Dict, List, Any

import fitz  # PyMuPDF
from langchain_core.documents import Document

try:
    from logger import get_logger
except ImportError:  # 以 src.kb_utils 方式从项目根目录导入时，src 不在 sys.path
    import logging
    get_logger = logging.getLogger

logger = get_logger(__name__)

# 四元组 JSON / 原始 CSV 的列序（与 test_samples.csv 表头一致）
QUAD_FEATURE_ORDER = [
    "is_FM", "is_RM", "is_DM", "is_HT", "is_WS", "is_NA",
    "FMWTARGETHOT", "RDWTARGETTOTAL", "PDIWIDTHTOL",
    "FMWIDTHACTHOT", "RMWIDTHACTHOT", "FMWTARGETCOL",
]


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

        # 过滤页眉/页脚
        filtered_blocks = []
        for b in text_blocks:
            _, y0, _, y1, _, _, _ = b
            if y0 < height * 0.1 or y1 > height * 0.9:
                continue
            filtered_blocks.append(b)
        text_blocks = filtered_blocks

        if not text_blocks:
            continue

        # 识别全宽块（跨栏标题、宽图注等）
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
            len(full_width_blocks) < 6
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
            sorted_blocks = sorted(normal_blocks, key=lambda b: b[1])
            body_text = "\n".join(b[4] for b in sorted_blocks)

        # 处理全宽块：顶部标题放正文前，底部注释放正文后
        if full_width_blocks:
            full_width_blocks.sort(key=lambda x: x[0])
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
                    "page": page_num + 1,
                },
            ))

    pdf_doc.close()
    return docs


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


def build_quadruple_texts(quad: Dict, scaler=None, feature_names: List[str] = None) -> Dict[str, Any]:
    """
    将单条四元组转为文本块和标准化特征向量。
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
        logger.warning("四元组故障特征解析失败: %s", feature_str)

    # 先重排为模型 feature_names 列序，再标准化（与查询端坐标系一致）
    if raw_vector and feature_names:
        raw_vector = reorder_features_to_model_order(raw_vector, feature_names)

    # 标准化特征向量（若 scaler 存在且向量有效）
    scaled_vector = []
    if raw_vector and scaler is not None:
        try:
            scaled_vector = scaler.transform([raw_vector])[0].tolist()
        except Exception as e:
            logger.warning("标准化失败: %s, 使用原始值", e)
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
        "诊断方案": solution,
    }
