# src/tools.py
"""
Agent 工具集 —— 将模型预测、知识检索、相似案例查询、特征重要性、批量CSV诊断
封装为 LangChain Tool，供 LangGraph Agent 调用。

返回值契约（所有工具统一）：
- 成功：JSON 字符串，含 "success": true 与业务字段
- 失败：JSON 字符串，含 success / error_code / message / recoverable / recommended_action，
  Agent 应按 recommended_action 恢复（追问用户 / 换角度重试 / 停止），不要原样复述错误
"""

import hashlib
import json
import os
import re
from collections import Counter
from typing import List

import pandas as pd
from langchain.tools import tool

from logger import get_logger
from memory_store import get_memory_store

logger = get_logger(__name__)
MEMORY = get_memory_store()

# 批量诊断返回给 LLM 的逐条详情上限（完整结果仍会写入长期记忆，可反查）
BATCH_DETAIL_LIMIT = int(os.getenv("RAG_BATCH_DETAIL_LIMIT", "50"))


# ==================== 统一返回值构造 ====================
def _ok(**fields) -> str:
    payload = {"success": True}
    payload.update(fields)
    return json.dumps(payload, ensure_ascii=False)


def _err(error_code: str, message: str, recoverable: bool = True,
         recommended_action: str = "") -> str:
    return json.dumps({
        "success": False,
        "error_code": error_code,
        "message": message,
        "recoverable": recoverable,
        "recommended_action": recommended_action,
    }, ensure_ascii=False)


def _parse_features(features) -> List[float]:
    """兼容 List[float] / List[int] / 逗号分隔字符串（含中文逗号），统一解析为浮点列表。"""
    if isinstance(features, str):
        parts = [x.strip() for x in features.replace("，", ",").split(",")]
        return [float(x) for x in parts if x]
    return [float(x) for x in features]


def create_tools(model_info, retriever, feature_names, predict_single, retrieve_context,
                 cached_retrieve_fn=None, batch_predict=None):
    """
    根据已加载的资源和函数构造 Agent 可用工具列表。
    """
    label_mapping = model_info['label_mapping']
    name_to_index = {name: idx for idx, name in label_mapping.items()}

    # 动态注入工具描述：特征顺序与已知故障类型（建工具时即可得，避免模型靠试错学习）
    feature_order_desc = "、".join(str(n) for n in feature_names)
    known_faults_desc = "、".join(str(v) for v in label_mapping.values())

    # ------------------------------------------------------------------
    # 工具 1：故障分类
    # ------------------------------------------------------------------
    @tool
    def classify_fault_tool(features: List[float]) -> str:
        """根据12维特征值预测故障类型（单卷诊断，一次会话只需调用一次）。
        输入: 长度为12的数字数组。返回: JSON（故障类型/故障描述/置信度）。
        结果为最终结论，获得后禁止再次调用本工具。
        不适用: 批量CSV诊断（用 batch_csv_diagnosis_tool）；用户未提供特征时应先追问，禁止编造数值。
        """
        try:
            feat_list = _parse_features(features)
            if len(feat_list) != 12:
                return _err(
                    "INVALID_FEATURE_COUNT",
                    f"需要恰好12个特征值，当前输入了 {len(feat_list)} 个。",
                    recoverable=True,
                    recommended_action="请向用户确认完整的12维特征值后重试。",
                )
            sample_df = pd.DataFrame([feat_list], columns=feature_names)
            diag = predict_single(model_info, sample_df)
            confidence = round(max(diag["probs"]), 4) if diag.get("probs") else None
            return _ok(
                **{
                    "故障类型": diag["fault_cn"],
                    "故障描述": diag["fault_desc"],
                    "置信度": confidence,
                    "诊断完成": True,
                }
            )
        except (ValueError, TypeError) as e:
            return _err(
                "INVALID_FEATURE_VALUE",
                f"特征值无法解析为数字: {e}",
                recoverable=True,
                recommended_action="请向用户确认特征值为纯数字后重试。",
            )
        except Exception as e:  # noqa: BLE001
            return _err(
                "PREDICT_FAILED",
                f"分类预测失败: {e}",
                recoverable=False,
                recommended_action="请告知用户模型服务暂不可用，建议稍后重试。",
            )

    # ------------------------------------------------------------------
    # 工具 2：知识库检索
    # ------------------------------------------------------------------
    @tool
    def retrieve_knowledge_tool(query: str) -> str:
        """在知识库中检索与查询相关的技术文档、故障定义和四元组知识。
        适合: 故障机理、处理方案、工艺知识类问题。
        不适合: 需要12维特征计算的诊断（用分类/相似案例工具）。
        """
        try:
            if cached_retrieve_fn is not None:
                context, _ = cached_retrieve_fn(query)
            else:
                context, _ = retrieve_context(retriever, query, feat_vec=None, fault_cn=None)
            if not context or not context.strip():
                return _err(
                    "NO_RESULTS",
                    f"知识库未返回与「{query}」直接匹配的内容。",
                    recoverable=True,
                    recommended_action=(
                        "可改用故障中文名称、故障机理关键词或调整方法等角度重新检索，最多重试一次；"
                        "仍无结果则基于已有知识回答。"
                    ),
                )
            n_hits = len(re.findall(r"^\[\d+\]", context, re.M))
            return _ok(n_hits=n_hits, knowledge=context)
        except Exception as e:  # noqa: BLE001
            return _err(
                "RETRIEVE_FAILED",
                f"知识检索失败: {e}",
                recoverable=False,
                recommended_action="请基于已有知识回答，或告知用户知识库暂不可用。",
            )

    # ------------------------------------------------------------------
    # 工具 3：相似历史案例查询
    # ------------------------------------------------------------------
    @tool
    def search_similar_cases_tool(features: List[float]) -> str:
        """根据12维特征向量搜索历史相似故障案例。
        输入: 长度为12的数字数组，顺序与 classify_fault_tool 一致。返回: JSON（相似案例及诊断方案）。
        """
        try:
            feat_list = _parse_features(features)
            if len(feat_list) != 12:
                return _err(
                    "INVALID_FEATURE_COUNT",
                    f"需要恰好12个特征值，当前输入了 {len(feat_list)} 个。",
                    recoverable=True,
                    recommended_action="请向用户确认完整的12维特征值后重试。",
                )
            scaler = model_info['scaler']
            feat_scaled = scaler.transform([feat_list])[0].tolist()
            _, sim_text = retrieve_context(retriever, "", feat_vec=feat_scaled)
            if not sim_text:
                return _err(
                    "NO_SIMILAR_CASES",
                    "未找到相似历史案例。",
                    recoverable=True,
                    recommended_action="可调用 retrieve_knowledge_tool 按故障类型检索处理方案。",
                )
            n_cases = sim_text.count("相似度：")
            return _ok(n_cases=n_cases, cases=sim_text)
        except (ValueError, TypeError) as e:
            return _err(
                "INVALID_FEATURE_VALUE",
                f"特征值无法解析为数字: {e}",
                recoverable=True,
                recommended_action="请向用户确认特征值为纯数字后重试。",
            )
        except Exception as e:  # noqa: BLE001
            return _err(
                "SIMILAR_SEARCH_FAILED",
                f"相似案例查询失败: {e}",
                recoverable=False,
                recommended_action="请告知用户相似案例检索暂不可用。",
            )

    # ------------------------------------------------------------------
    # 工具 4：特征重要性查询
    # ------------------------------------------------------------------
    @tool
    def get_feature_importance_tool(fault_type: str) -> str:
        """查询指定故障类型的关键特征重要性排名（基于MIC注意力权重）。
        输入: 故障类型中文名（枚举值，见工具描述）。返回: JSON（前5个关键特征及权重）。
        """
        try:
            model = model_info["model"]
            if hasattr(model, "attention"):
                attn_weights = model.attention.get_attention_matrix()
            elif hasattr(model, "models"):
                attn_weights = model.models[0].attention.get_attention_matrix()
            else:
                return _err(
                    "UNSUPPORTED",
                    "当前模型不支持特征重要性查询。",
                    recoverable=False,
                    recommended_action="请直接基于分类结果与知识检索回答。",
                )

            if fault_type not in name_to_index:
                return _err(
                    "UNKNOWN_FAULT_TYPE",
                    f"未找到故障类型 '{fault_type}'。",
                    recoverable=True,
                    recommended_action=f"请从已知故障类型中选择后重试: {known_faults_desc}",
                )
            cls_idx = name_to_index[fault_type]
            feat_imp = attn_weights[cls_idx]
            named_imp = [
                (str(feature_names[i]), round(float(feat_imp[i]), 4))
                for i in range(len(feature_names))
            ]
            named_imp.sort(key=lambda x: x[1], reverse=True)
            return _ok(特征重要性=named_imp[:5])
        except Exception as e:  # noqa: BLE001
            return _err(
                "IMPORTANCE_FAILED",
                f"获取特征重要性失败: {e}",
                recoverable=False,
                recommended_action="请直接基于分类结果与知识检索回答。",
            )

    # ------------------------------------------------------------------
    # 工具 5：批量 CSV 诊断
    # ------------------------------------------------------------------
    @tool
    def batch_csv_diagnosis_tool(csv_path: str) -> str:
        """对上传的CSV文件进行批量宽度缺陷诊断（CSV 场景必须优先调用本工具，禁止逐条调用分类工具）。
        输入: CSV文件的本地路径。返回: JSON（total_samples/distribution/details），
        details 每条含 features_named（全部12维特征的带名字典，含 FMWTARGETHOT/RDWTARGETTOTAL/
        PDIWIDTHTOL/FMWIDTHACTHOT/RMWIDTHACTHOT/FMWTARGETCOL 六个宽度值与六个 is_* 波谷标志位，
        含义见特征定义表）与 features（同数据的12维数组，仅为兼容保留）；
        解读任何特征值必须以 features_named 为准，禁止按下标数 features 数组。
        details 在样本较多时会截断，完整逐卷结果已写入长期记忆，可用 query_hist_diag_tool 反查。
        一次会话只需调用一次。
        """
        try:
            if not batch_predict:
                return _err(
                    "NOT_CONFIGURED",
                    "批量预测功能未配置。",
                    recoverable=False,
                    recommended_action="请告知用户批量诊断暂不可用。",
                )
            df = pd.read_csv(csv_path)
            missing = [c for c in feature_names if c not in df.columns]
            if missing:
                return _err(
                    "MISSING_COLUMNS",
                    f"CSV 文件缺少必要列：{', '.join(missing)}",
                    recoverable=True,
                    recommended_action=f"请告知用户 CSV 需包含列: {feature_order_desc}",
                )

            diag_list = batch_predict(model_info, df[feature_names])

            fault_counts = Counter(d['fault_cn'] for d in diag_list)

            full_results = []
            for i, d in enumerate(diag_list):
                feats = df.iloc[i].tolist()
                row = df.iloc[i]
                full_results.append({
                    "index": i,
                    "features": feats,
                    # 12 维特征带字段名字典，按列名取值
                    # （模型 feature_names 与 CSV 列序可能不同，禁止按下标 zip 对齐）
                    "features_named": {
                        name: (int(round(float(row[name])))
                               if str(name).startswith("is_") else float(row[name]))
                        for name in feature_names
                    },
                    "fault_cn": d['fault_cn'],
                    "fault_desc": d['fault_desc'],
                    "confidence": round(max(d['probs']), 4) if d.get('probs') else None,
                })

            # 完整逐卷结果写入长期记忆（截断不影响可追溯性），与历史查询工具共用同一索引
            for d in full_results:
                feats = d["features"]
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

            truncated = len(full_results) > BATCH_DETAIL_LIMIT
            payload = {
                "total_samples": len(diag_list),
                "distribution": dict(fault_counts.most_common()),
                "details": full_results[:BATCH_DETAIL_LIMIT],
                "details_truncated": truncated,
            }
            if truncated:
                payload["details_note"] = (
                    f"details 仅保留前 {BATCH_DETAIL_LIMIT} 条（共 {len(full_results)} 条），"
                    "完整结果已写入长期记忆，可用 query_hist_diag_tool 按特征反查。"
                )
            return _ok(**payload)
        except FileNotFoundError:
            return _err(
                "FILE_NOT_FOUND",
                f"CSV 文件不存在: {csv_path}",
                recoverable=True,
                recommended_action="请向用户确认文件已上传，或让用户重新上传。",
            )
        except Exception as e:  # noqa: BLE001
            return _err(
                "BATCH_FAILED",
                f"批量CSV诊断失败: {e}",
                recoverable=False,
                recommended_action="请告知用户批量诊断失败，建议检查 CSV 格式后重试。",
            )

    # ------------------------------------------------------------------
    # 工具 6：历史诊断查询（跨会话追溯）
    # ------------------------------------------------------------------
    @tool
    def query_hist_diag_tool(features: List[float]) -> str:
        """根据12维特征值查询该特征向量的历史诊断记录（跨会话追溯）。
        输入: 长度为12的数字数组，顺序与 classify_fault_tool 一致。返回: JSON（历史诊断记录）。
        """
        try:
            feat_list = _parse_features(features)
            if len(feat_list) != 12:
                return _err(
                    "INVALID_FEATURE_COUNT",
                    f"需要恰好12个特征值，当前输入了 {len(feat_list)} 个。",
                    recoverable=True,
                    recommended_action="请向用户确认完整的12维特征值后重试。",
                )
            rec = MEMORY.get_diag_by_features(feat_list)
            if not rec:
                return _err(
                    "NO_HISTORY",
                    "未找到该特征向量的历史诊断记录。",
                    recoverable=True,
                    recommended_action="该卷可能尚未诊断过：如需诊断可调用 classify_fault_tool。",
                )
            return _ok(
                fault_cn=rec.get("fault_cn"),
                fault_desc=rec.get("fault_desc"),
                confidence=rec.get("confidence"),
                features=rec.get("features"),
                updated_at=rec.get("updated_at"),
            )
        except (ValueError, TypeError) as e:
            return _err(
                "INVALID_FEATURE_VALUE",
                f"特征值无法解析为数字: {e}",
                recoverable=True,
                recommended_action="请向用户确认特征值为纯数字后重试。",
            )
        except Exception as e:  # noqa: BLE001
            return _err(
                "HISTORY_QUERY_FAILED",
                f"历史诊断查询失败: {e}",
                recoverable=False,
                recommended_action="请告知用户历史查询暂不可用。",
            )

    # 动态补充描述（特征顺序/故障类型清单建工具时才可得）
    classify_fault_tool.description += f"\n特征顺序（必须一致）: {feature_order_desc}"
    search_similar_cases_tool.description += f"\n特征顺序（必须一致）: {feature_order_desc}"
    query_hist_diag_tool.description += f"\n特征顺序（必须一致）: {feature_order_desc}"
    get_feature_importance_tool.description += f"\n已知故障类型（枚举）: {known_faults_desc}"

    return [
        classify_fault_tool,
        retrieve_knowledge_tool,
        search_similar_cases_tool,
        get_feature_importance_tool,
        batch_csv_diagnosis_tool,
        query_hist_diag_tool,
    ]
