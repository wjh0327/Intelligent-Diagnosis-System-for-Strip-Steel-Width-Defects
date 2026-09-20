# -*- coding: utf-8 -*-
"""
src/hybrid_model.py —— 论文主模型（双分支混合CNN）推理服务层
==============================================================
替代原 model_def.py 的 MIC 注意力集成模型 (MICAN)。

模型: models/cnn_hybrid_diag.pt —— 论文主模型（单种子）
  - 曲线分支: 三段全长宽度曲线 RM/FM/DC ＋ 位置编码（4通道×128点，按目标宽度归一）
  - 特征分支: 12维机理特征（6工艺参数 ＋ 6规则flag，由 features.py 按原版 te.py 逻辑现算）
输出: 七类概率 ＋ 置信度门限（最高概率<0.6 或 前两类差<0.2 → 建议人工复核，论文3.5节）
      ＋ 梯度×输入归因证据（特征Top3 与 曲线通道关注度，替代原注意力权重）
"""
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch

from cnn_model import (HybridWidthCNN, build_curve_tensor,
                       gradient_attribution_hybrid, CLASS_ORDER)
from features import compute_flags_original, BASE_COLS, FLAG_COLS

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR.parent / "models"
CKPT_PATH = str(MODEL_DIR / "cnn_hybrid_diag.pt")
IMPORTANCE_PATH = str(MODEL_DIR / "cnn_hybrid_importance.json")

# 诊断输入必需的原始列（STRIPNO/label 可选；6个 is_* flag 由本服务现算，无需提供）
REQUIRED_COLUMNS = [
    'FMWTARGETHOT', 'RDWTARGETTOTAL', 'PDIWIDTHTOL',
    'FMWIDTHACTHOT', 'RMWIDTHACTHOT', 'FMWTARGETCOL',
    'RMDATASEQ', 'FMDATASEQ', 'DCDATASEQ',
]
FEAT12 = BASE_COLS + FLAG_COLS

ROOT_CAUSE = {
    '卷取拉钢': ('卷取咬入/脱离时张力突变', '检查卷取机张力闭环响应'),
    '整体宽':   ('自然宽展偏差或PDI不当', '侧导板开口度设定'),
    '整体窄':   ('自然宽展偏差或PDI不当', '侧导板开口度设定'),
    '楔形':     ('横向不均匀延伸', '两侧辊缝对称性'),
    '没有拉钢': ('轧制过程正常', '复核确认即可'),
    '粗轧头尾': ('短行程控制不合理', '粗轧头尾SSC策略'),
    '精轧拉钢': ('活套张力控制不稳定', '精轧机架间活套'),
}
LABEL_MAPPING = {i: name for i, name in enumerate(CLASS_ORDER)}
PROBLEM_DESCRIPTIONS = {fault: desc for fault, (desc, _) in ROOT_CAUSE.items()}
FEAT_CN = {
    'FMWTARGETHOT': '精轧目标宽(热)', 'RDWTARGETTOTAL': '粗轧目标宽', 'PDIWIDTHTOL': 'PDI宽度公差',
    'FMWIDTHACTHOT': '精轧实际宽(热)', 'RMWIDTHACTHOT': '粗轧实际宽(热)', 'FMWTARGETCOL': '精轧目标宽(冷)',
    'is_FM': '精轧波谷', 'is_RM': '粗轧波谷', 'is_DM': '卷取波谷',
    'is_HT': '头尾同位', 'is_WS': '中部不均', 'is_NA': '整体窄',
}

_state = None


def load_model() -> Dict[str, Any]:
    """惰性加载模型与按类归因缓存（service.init_resources 保证只加载一次）。"""
    global _state
    if _state is None:
        ckpt = torch.load(CKPT_PATH, weights_only=False)
        model = HybridWidthCNN(in_channels=ckpt['in_channels'], n_aux=len(ckpt['feat_cols']))
        model.load_state_dict(ckpt['state_dict'])
        model.eval()
        importance: Dict[str, List] = {}
        if os.path.exists(IMPORTANCE_PATH):
            with open(IMPORTANCE_PATH, encoding='utf-8') as f:
                importance = json.load(f)
        _state = {'model': model, 'ckpt': ckpt, 'importance': importance}
    return _state


def make_aux(df: pd.DataFrame) -> np.ndarray:
    """12维机理特征: 6个工艺参数直接取列, 6个规则flag按 te.py 原版逻辑逐条现算。"""
    flags = df.apply(compute_flags_original, axis=1, result_type='expand')
    flags.columns = FLAG_COLS
    aux = pd.concat([df[BASE_COLS].reset_index(drop=True),
                     flags.reset_index(drop=True)], axis=1)
    return aux[FEAT12].values.astype(np.float32)


def importance_top(fault_type: str, k: int = 5) -> List[List[Any]]:
    """指定故障类型的特征重要性 Top-k（预计算梯度归因，训练集按类聚合）。"""
    state = load_model()
    pairs = state['importance'].get(fault_type, [])
    return [[FEAT_CN.get(f, f), round(float(w), 4)] for f, w in pairs[:k]]


def record_to_feat12(record: Dict[str, Any]) -> List[float]:
    """单条带钢记录(dict) → 12维机理特征向量（供相似案例检索与历史哈希）。"""
    df = pd.DataFrame([record])
    flags = df.apply(compute_flags_original, axis=1, result_type='expand')
    flags.columns = FLAG_COLS
    aux = pd.concat([df[BASE_COLS].reset_index(drop=True),
                     flags.reset_index(drop=True)], axis=1)
    return aux[FEAT12].iloc[0].tolist()


def strip_hash(record: Dict[str, Any], feat12: List[float]) -> str:
    """历史追溯键: md5(STRIPNO ＋ 12维特征)（迁移决策二，表结构不变）。
    前6个工艺参数按3位小数、6个flag按整数参与哈希，保证浮点表示差异（如
    批量返回值经界面/LLM往返后的舍入）不影响同一条带钢的键一致性。"""
    stripno = str(record.get('STRIPNO', ''))
    vals = [round(float(f), 3) if i < 6 else int(round(float(f)))
            for i, f in enumerate(feat12)]
    payload = json.dumps([stripno] + vals, ensure_ascii=False)
    return hashlib.md5(payload.encode('utf-8')).hexdigest()


def scale_feat12(feat12: List[float]) -> List[float]:
    """12维特征标准化（用训练集统计量），供 Milvus 相似案例检索。"""
    state = load_model()
    ckpt = state['ckpt']
    mean = np.asarray(ckpt['feat_mean'], dtype=np.float32)
    std = np.asarray(ckpt['feat_std'], dtype=np.float32)
    std[std < 1e-8] = 1.0
    return ((np.asarray(feat12, dtype=np.float32) - mean) / std).tolist()


def predict_strips(df: pd.DataFrame, with_evidence: bool = True) -> List[Dict[str, Any]]:
    """对原始带钢记录 DataFrame 批量预测。

    输入: df 须含 REQUIRED_COLUMNS 列（STRIPNO/label 可选）。
    防御: 工艺参数或曲线缺失的行标记为'数据不完整'并转人工复核，不参与预测。
    返回: 逐条诊断结果列表（pred_class/fault_cn/fault_desc/probs/confidence/
          needs_review/evidence/evidence_named/curve_focus/root_cause/check_item/
          feat_vector12/flags_named）。
    """
    state = load_model()
    model, ckpt = state['model'], state['ckpt']
    fmean = np.asarray(ckpt['feat_mean'], dtype=np.float32)
    fstd = np.asarray(ckpt['feat_std'], dtype=np.float32)
    fstd[fstd < 1e-8] = 1.0

    Xc_all = build_curve_tensor(df)
    Xf_all = make_aux(df)
    valid = (np.isfinite(Xf_all).all(axis=1)
             & np.isfinite(Xc_all).all(axis=(1, 2)))
    vi = list(np.where(valid)[0])

    results: List[Dict[str, Any]] = []
    for gi in range(len(df)):
        results.append({
            'stripno': str(df.iloc[gi].get('STRIPNO', '')),
            'pred_class': -1,
            'fault_cn': '数据不完整',
            'fault_desc': '工艺参数或曲线数据存在缺失，无法自动诊断',
            'probs': None,
            'confidence': None,
            'needs_review': True,
            'evidence': '',
            'evidence_named': [],
            'curve_focus': '',
            'root_cause': '',
            'check_item': '',
            'feat_vector12': [float(v) for v in Xf_all[gi]] if np.isfinite(Xf_all[gi]).all() else [],
            'flags_named': {},
        })
    if not vi:
        return results

    Xc = torch.FloatTensor(Xc_all[valid])
    Xf = torch.FloatTensor((Xf_all[valid] - fmean) / fstd)
    with torch.no_grad():
        proba = torch.softmax(model(Xc, Xf), dim=1).numpy()
    pred = proba.argmax(1)
    conf = proba.max(1)
    srt = np.sort(proba, axis=1)
    gap = srt[:, -1] - srt[:, -2]
    review = (conf < 0.6) | (gap < 0.2)

    aux_attr, curve_attr, _ = (None, None, None)
    if with_evidence:
        aux_attr, curve_attr, _ = gradient_attribution_hybrid(model, Xc, Xf, fmean, fstd)

    for local_i, gi in enumerate(vi):
        p = int(pred[local_i])
        fault_cn = CLASS_ORDER[p]
        row = df.iloc[gi]
        rec = results[gi]
        rec.update({
            'pred_class': p,
            'fault_cn': fault_cn,
            'fault_desc': PROBLEM_DESCRIPTIONS.get(fault_cn, ''),
            'probs': [round(float(v), 6) for v in proba[local_i]],
            'confidence': round(float(conf[local_i]), 4),
            'needs_review': bool(review[local_i]),
            'root_cause': ROOT_CAUSE.get(fault_cn, ('', ''))[0],
            'check_item': ROOT_CAUSE.get(fault_cn, ('', ''))[1],
            'feat_vector12': [float(v) for v in Xf_all[gi]],
            'flags_named': {
                name: int(round(float(row[name])))
                for name in FLAG_COLS if name in row
            },
        })
        if with_evidence:
            s = pd.Series(aux_attr[local_i], index=FEAT12)
            top = s.reindex(s.abs().sort_values(ascending=False).head(3).index)
            rec['evidence_named'] = [
                {'feature': FEAT_CN.get(k, k), 'value': round(float(v), 3)}
                for k, v in top.items()
            ]
            rec['evidence'] = '；'.join(
                f'{FEAT_CN.get(k, k)}({v:+.2f})' for k, v in top.items()
            )
            ch = ['粗轧', '精轧', '卷取'][int(np.abs(curve_attr[local_i]).argmax())]
            rec['curve_focus'] = ch
            rec['evidence'] += f'｜曲线关注:{ch}段'
    return results
