# src/tools.py
"""
Agent 工具集 —— 将模型预测、知识检索、相似案例查询、特征重要性、批量CSV诊断
封装为 LangChain Tool，供 LangGraph Agent 调用。
"""

import json
import numpy as np
import pandas as pd
from langchain.tools import tool
from typing import List, Dict, Any


def create_tools(model_info, retriever, feature_names, predict_single, retrieve_context,
                 cached_retrieve_fn=None, batch_predict=None):
    """
    根据已加载的资源和函数构造 Agent 可用工具列表。
    """
    label_mapping = model_info['label_mapping']
    name_to_index = {name: idx for idx, name in label_mapping.items()}

    # ------------------------------------------------------------------
    # 工具 1：故障分类
    # ------------------------------------------------------------------
    @tool
    def classify_fault_tool(features: str) -> str:
        """
        根据12维特征值预测故障类型。
        输入格式: 逗号分隔的12个浮点数，例如 "0.5,1.2,0.8,..."
        返回: 包含故障中文名、描述及置信度的JSON字符串。
        """
        try:
            feat_list = [float(x.strip()) for x in features.split(",")]
            if len(feat_list) != 12:
                return f"错误：需要恰好12个特征值，当前输入了 {len(feat_list)} 个。"
            sample_df = pd.DataFrame([feat_list], columns=feature_names)
            diag = predict_single(model_info, sample_df)
            confidence = round(max(diag["probs"]), 4) if diag.get("probs") else None
            return json.dumps(
                {
                    "故障类型": diag["fault_cn"],
                    "故障描述": diag["fault_desc"],
                    "置信度": confidence,
                },
                ensure_ascii=False,
            )
        except Exception as e:
            return f"分类预测失败: {str(e)}"

    # ------------------------------------------------------------------
    # 工具 2：知识库检索
    # ------------------------------------------------------------------
    @tool
    def retrieve_knowledge_tool(query: str) -> str:
        """在知识库中检索与查询相关的技术文档、故障定义和四元组知识。"""
        try:
            if cached_retrieve_fn is not None:
                context, _ = cached_retrieve_fn(query)
            else:
                context, _ = retrieve_context(retriever, query, feat_vec=None, fault_cn=None)
            return context
        except Exception as e:
            return f"知识检索失败: {str(e)}"

    # ------------------------------------------------------------------
    # 工具 3：相似历史案例查询
    # ------------------------------------------------------------------
    @tool
    def search_similar_cases_tool(features: str) -> str:
        """根据12维特征向量搜索历史相似故障案例，输入为逗号分隔的原始特征值"""
        try:
            feat_list = [float(x.strip()) for x in features.split(",")]
            if len(feat_list) != 12:
                return f"错误：需要恰好12个特征值，当前输入了 {len(feat_list)} 个。"

            scaler = model_info['scaler']
            feat_scaled = scaler.transform([feat_list])[0].tolist()

            _, sim_text = retrieve_context(retriever, "", feat_vec=feat_scaled)
            return sim_text if sim_text else "未找到相似案例。"
        except Exception as e:
            return f"相似案例查询失败: {str(e)}"

    # ------------------------------------------------------------------
    # 工具 4：特征重要性查询
    # ------------------------------------------------------------------
    @tool
    def get_feature_importance_tool(fault_type: str) -> str:
        """查询指定故障类型的关键特征重要性排名（基于MIC注意力权重）"""
        try:
            model = model_info["model"]
            if hasattr(model, "attention"):
                attn_weights = model.attention.get_attention_matrix()
            elif hasattr(model, "models"):
                attn_weights = model.models[0].attention.get_attention_matrix()
            else:
                return "当前模型不支持特征重要性查询。"

            if fault_type not in name_to_index:
                known_types = ', '.join(name_to_index.keys())
                return f"未找到故障类型 '{fault_type}'，已知类型: {known_types}"
            cls_idx = name_to_index[fault_type]

            feat_imp = attn_weights[cls_idx]
            named_imp = [
                (str(feature_names[i]), round(float(feat_imp[i]), 4))
                for i in range(len(feature_names))
            ]
            named_imp.sort(key=lambda x: x[1], reverse=True)
            return json.dumps(named_imp[:5], ensure_ascii=False)
        except Exception as e:
            return f"获取特征重要性失败: {str(e)}"

    # ------------------------------------------------------------------
    # 工具 5：批量 CSV 诊断
    # ------------------------------------------------------------------
    @tool
    def batch_csv_diagnosis_tool(csv_path: str) -> str:
        """
        对上传的CSV文件进行批量宽度缺陷诊断，返回所有样本的完整诊断结果（JSON格式）。
        输入: CSV文件的本地路径。
        返回: 包含所有样本诊断详情的JSON字符串。
        """
        try:
            if not batch_predict:
                return json.dumps({"error": "批量预测功能未配置。"}, ensure_ascii=False)

            df = pd.read_csv(csv_path)
            missing = [c for c in feature_names if c not in df.columns]
            if missing:
                return json.dumps({"error": f"CSV 文件缺少必要列：{', '.join(missing)}"}, ensure_ascii=False)

            diag_list = batch_predict(model_info, df[feature_names])

            # 统计故障分布
            from collections import Counter
            fault_counts = Counter(d['fault_cn'] for d in diag_list)

            # 构建完整结果（包含所有样本的详细信息）
            full_results = []
            for i, d in enumerate(diag_list):
                full_results.append({
                    "index": i,
                    "features": df.iloc[i].tolist(),  # 原始特征值
                    "fault_cn": d['fault_cn'],
                    "fault_desc": d['fault_desc'],
                    "confidence": round(max(d['probs']), 4) if d.get('probs') else None
                })

            result_obj = {
                "total_samples": len(diag_list),
                "distribution": dict(fault_counts.most_common()),
                "details": full_results  # 全部样本详情
            }

            return json.dumps(result_obj, ensure_ascii=False)
        except Exception as e:
            return json.dumps({"error": f"批量CSV诊断失败: {str(e)}"}, ensure_ascii=False)

    return [
        classify_fault_tool,
        retrieve_knowledge_tool,
        search_similar_cases_tool,
        get_feature_importance_tool,
        batch_csv_diagnosis_tool,
    ]