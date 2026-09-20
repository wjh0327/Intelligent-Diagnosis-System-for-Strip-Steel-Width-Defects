# src/tools.py
"""
Agent 工具集 —— 将模型预测、知识检索、相似案例查询、特征重要性、批量CSV诊断、历史追溯
封装为 LangChain Tool，供 LangGraph Agent 调用。

v2（论文主模型：双分支混合CNN）：
- 诊断类工具的输入契约从「12维特征向量」升级为「单条带钢原始记录」：
  STRIPNO(可选) + 6个工艺参数 + 3段全长宽度曲线（与 data/width.csv 行格式一致）；
  6个 is_* 规则标志位由模型服务按原版机理逻辑自动计算，无需提供。
- 特征重要性从 MIC 注意力权重改为 训练集梯度归因（按类聚合）。
- 相似案例检索与历史追溯仍以 12 维机理特征为索引键（对 LLM 透明）。

返回值契约（所有工具统一）：
- 成功：JSON 字符串，含 "success": true 与业务字段
- 失败：JSON 字符串，含 success / error_code / message / recoverable / recommended_action，
  Agent 应按 recommended_action 恢复（追问用户 / 换角度重试 / 停止），不要原样复述错误
"""

import json
import os
import re
from collections import Counter
from typing import List

import pandas as pd
from langchain.tools import tool

from features import FEAT12, compute_flags_original
from logger import get_logger
from memory_store import get_memory_store

logger = get_logger(__name__)
MEMORY = get_memory_store()

# 批量诊断返回给 LLM 的逐条详情上限（完整结果仍会写入长期记忆，可反查）
BATCH_DETAIL_LIMIT = int(os.getenv("RAG_BATCH_DETAIL_LIMIT", "50"))

# 单条带钢记录的输入格式说明（注入工具描述，避免模型靠试错学习）
STRIP_FORMAT_DESC = (
    "JSON对象，字段: STRIPNO(可选) + 6个工艺参数 "
    "(FMWTARGETHOT/RDWTARGETTOTAL/PDIWIDTHTOL/FMWIDTHACTHOT/RMWIDTHACTHOT/FMWTARGETCOL) "
    "+ 3段全长宽度曲线 (RMDATASEQ/FMDATASEQ/DCDATASEQ，逗号分隔数值字符串)"
)

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


def _parse_strip_json(strip_json: str):
    """解析单条带钢记录 JSON，并校验必需字段。返回 (record_dict, None) 或 (None, err_str)。"""
    try:
        record = json.loads(strip_json)
    except json.JSONDecodeError as e:
        return None, _err(
            "INVALID_JSON",
            f"无法解析带钢记录 JSON: {e}",
            recoverable=True,
            recommended_action="请检查 JSON 格式后重试；字段说明见工具描述。",
        )
    if not isinstance(record, dict):
        return None, _err(
            "INVALID_RECORD",
            "带钢记录必须是 JSON 对象（字段: STRIPNO + 6工艺参数 + 3段曲线）。",
            recoverable=True,
            recommended_action="请按工具描述补全字段后重试。",
        )
    return record, None


def _check_missing(record: dict, required_columns: List[str]):
    missing = [c for c in required_columns if c not in record]
    if missing:
        return _err(
            "MISSING_FIELDS",
            f"带钢记录缺少必要字段: {', '.join(missing)}",
            recoverable=True,
            recommended_action=f"请补全字段后重试。必需字段: {', '.join(required_columns)}",
        )
    return None


def _feat12_from_record(record: dict):
    """从带钢记录提取12维机理特征，支持两种模式：
    - 完整模式: 含3段曲线(RMDATASEQ/FMDATASEQ/DCDATASEQ) → 按机理逻辑现算6个flag
    - 紧凑模式: 已含6个 is_* 值（如批量诊断返回的 features_named）→ 直接组装
    返回 (feat12_list, None) 或 (None, err_str)。"""
    has_flags = all(k in record for k in ('is_FM', 'is_RM', 'is_DM', 'is_HT', 'is_WS', 'is_NA'))
    has_curves = all(k in record for k in ('RMDATASEQ', 'FMDATASEQ', 'DCDATASEQ'))
    missing_params = [c for c in (
        'FMWTARGETHOT', 'RDWTARGETTOTAL', 'PDIWIDTHTOL',
        'FMWIDTHACTHOT', 'RMWIDTHACTHOT', 'FMWTARGETCOL') if c not in record]
    if missing_params:
        return None, _err(
            'MISSING_FIELDS',
            f'带钢记录缺少工艺参数字段: {", ".join(missing_params)}',
            recoverable=True,
            recommended_action='请补全 6 个工艺参数后重试。',
        )
    if has_flags:
        feat12 = [float(record[c]) for c in (
            'FMWTARGETHOT', 'RDWTARGETTOTAL', 'PDIWIDTHTOL',
            'FMWIDTHACTHOT', 'RMWIDTHACTHOT', 'FMWTARGETCOL')]
        feat12 += [float(record[c]) for c in ('is_FM', 'is_RM', 'is_DM', 'is_HT', 'is_WS', 'is_NA')]
        return feat12, None
    if has_curves:
        df = pd.DataFrame([record])
        flags = df.apply(compute_flags_original, axis=1, result_type='expand')
        feat12 = [float(df.iloc[0][c]) for c in (
            'FMWTARGETHOT', 'RDWTARGETTOTAL', 'PDIWIDTHTOL',
            'FMWIDTHACTHOT', 'RMWIDTHACTHOT', 'FMWTARGETCOL')]
        feat12 += [float(flags.iloc[0][c]) for c in ('is_FM', 'is_RM', 'is_DM', 'is_HT', 'is_WS', 'is_NA')]
        return feat12, None
    return None, _err(
        'MISSING_FIELDS',
        '带钢记录需含 3 段曲线（完整模式）或 6 个 is_* 标志位（紧凑模式，如批量诊断返回的 features_named）。',
        recoverable=True,
        recommended_action='请补全曲线列或改用批量诊断结果中的 features_named 字段。',
    )


def create_tools(model_info, retriever, feature_names, predict_single, retrieve_context,
                 cached_retrieve_fn=None, batch_predict=None):
    """
    根据已加载的资源和函数构造 Agent 可用工具列表。

    feature_names 为诊断必需的原始带钢列（6工艺参数 + 3段曲线）。
    model_info 额外携带: record_to_feat12 / scale_feat12 / strip_hash / importance_cache。
    """
    known_faults_desc = "、".join(
        str(v) for _, v in sorted(model_info['label_mapping'].items())
    )
    required_desc = "、".join(str(c) for c in feature_names)

    # ------------------------------------------------------------------
    # 工具 1：单条带钢诊断
    # ------------------------------------------------------------------
    @tool
    def diagnose_strip_tool(strip_json: str) -> str:
        """根据单条带钢的原始数据预测宽度缺陷类型（单卷诊断，一次会话只需调用一次）。
        输入: 单条带钢记录的 JSON 字符串（格式见字段说明）。
        返回: JSON（故障类型/故障描述/置信度/需人工复核/归因证据/根因/检查项）。
        结果为最终结论，获得后禁止再次调用本工具。
        不适用: 批量CSV诊断（用 batch_csv_diagnosis_tool）；用户未提供数据时应先追问，
        禁止编造数值或曲线。
        """
        try:
            record, err = _parse_strip_json(strip_json)
            if err:
                return err
            err = _check_missing(record, feature_names)
            if err:
                return err
            sample_df = pd.DataFrame([record])
            diag = predict_single(model_info, sample_df)
            return _ok(
                **{
                    "故障类型": diag["fault_cn"],
                    "故障描述": diag["fault_desc"],
                    "置信度": diag["confidence"],
                    "需人工复核": "是" if diag.get("needs_review") else "否",
                    "归因证据": diag.get("evidence", ""),
                    "根因": diag.get("root_cause", ""),
                    "检查项": diag.get("check_item", ""),
                    "诊断完成": True,
                }
            )
        except Exception as e:  # noqa: BLE001
            return _err(
                "PREDICT_FAILED",
                f"诊断预测失败: {e}",
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
        不适合: 需要结合带钢曲线数据计算的诊断（用诊断/相似案例工具）。
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
    def search_similar_cases_tool(strip_json: str) -> str:
        """根据单条带钢的原始数据搜索历史相似故障案例。
        输入: JSON记录，两种方式任选 —— ①完整记录（6工艺参数+3段曲线，自动现算flag）；②紧凑记录（6工艺参数+6个is_*值，如批量诊断返回的 features_named）。
        返回: JSON（相似案例及诊断方案）。
        """
        try:
            record, err = _parse_strip_json(strip_json)
            if err:
                return err
            feat12, err = _feat12_from_record(record)
            if err:
                return err
            feat_scaled = model_info['scale_feat12'](feat12)
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
        """查询指定故障类型的关键特征重要性排名（基于训练集梯度归因，按类聚合）。
        输入: 故障类型中文名（枚举值，见工具描述）。返回: JSON（前5个关键特征及归因权重）。
        """
        try:
            imp = model_info['importance_cache'].get(fault_type)
            if imp is None:
                return _err(
                    "UNKNOWN_FAULT_TYPE",
                    f"未找到故障类型 '{fault_type}'。",
                    recoverable=True,
                    recommended_action=f"请从已知故障类型中选择后重试: {known_faults_desc}",
                )
            named_imp = [(str(n), round(float(w), 4)) for n, w in imp[:5]]
            return _ok(特征重要性=named_imp)
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
        """对上传的CSV文件进行批量宽度缺陷诊断（CSV 场景必须优先调用本工具，禁止逐条调用诊断工具）。
        CSV 格式: 与 width.csv 一致 —— STRIPNO(可选) + 6个工艺参数 + 3段全长宽度曲线列；
        6个 is_* 规则标志位无需提供（服务自动按机理逻辑计算）。
        返回: JSON（total_samples/distribution/needs_review_count/details），
        details 每条含 stripno/故障类型/置信度/需人工复核/归因证据/12维机理特征带名字典(features_named，
        含义见特征定义表)；details 在样本较多时会截断，完整逐卷结果已写入长期记忆，
        可用 query_hist_diag_tool 反查。一次会话只需调用一次。
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
                    recommended_action=f"请告知用户 CSV 需包含列: {required_desc}（is_* 标志位无需提供）。",
                )

            diag_list = batch_predict(model_info, df)

            fault_counts = Counter(d['fault_cn'] for d in diag_list)
            review_count = sum(1 for d in diag_list if d.get('needs_review'))

            full_results = []
            for i, d in enumerate(diag_list):
                row = df.iloc[i]
                feats12 = d.get('feat_vector12', [])
                full_results.append({
                    "index": i,
                    "stripno": d.get('stripno', ''),
                    # 12 维机理特征带字段名字典（6参数原始值 + 6个规则标志位按机理现算）
                    "features_named": {
                        name: (int(round(float(feats12[j])))
                               if str(name).startswith("is_") else round(float(feats12[j]), 3))
                        for j, name in enumerate(FEAT12)
                    },
                    "fault_cn": d['fault_cn'],
                    "fault_desc": d['fault_desc'],
                    "confidence": d.get('confidence'),
                    "needs_review": "是" if d.get('needs_review') else "否",
                    "归因证据": d.get('evidence', ''),
                })

            # 完整逐卷结果写入长期记忆（截断不影响可追溯性），与历史查询工具共用同一索引
            for i, d in enumerate(diag_list):
                record = df.iloc[i].to_dict()
                feats12 = d.get('feat_vector12', [])
                if feats12:
                    feat_hash = model_info['strip_hash'](record, feats12)
                    MEMORY.save_diag(feat_hash, {
                        "fault_cn": d.get("fault_cn"),
                        "fault_desc": d.get("fault_desc"),
                        "confidence": d.get("confidence"),
                        "features": feats12,
                        "stripno": d.get("stripno", ""),
                        "needs_review": d.get("needs_review", False),
                        "evidence": d.get("evidence", ""),
                    })

            truncated = len(full_results) > BATCH_DETAIL_LIMIT
            payload = {
                "total_samples": len(diag_list),
                "distribution": dict(fault_counts.most_common()),
                "needs_review_count": review_count,
                "details": full_results[:BATCH_DETAIL_LIMIT],
                "details_truncated": truncated,
            }
            if truncated:
                payload["details_note"] = (
                    f"details 仅保留前 {BATCH_DETAIL_LIMIT} 条（共 {len(full_results)} 条），"
                    "完整结果已写入长期记忆，可用 query_hist_diag_tool 反查。"
                )
            if review_count:
                payload["review_note"] = (
                    f"{review_count} 条样本置信度不足，已标记需人工复核；"
                    "解读时请对这些样本提示建议人工确认，不要给出确定性结论。"
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
    def query_hist_diag_tool(strip_json: str) -> str:
        """根据单条带钢的原始数据查询其历史诊断记录（跨会话追溯）。
        输入: JSON记录（同 search_similar_cases_tool 的两种方式），建议附带批量诊断结果中的 STRIPNO 以精确匹配历史键。
        返回: JSON（历史诊断记录）。
        """
        try:
            record, err = _parse_strip_json(strip_json)
            if err:
                return err
            feat12, err = _feat12_from_record(record)
            if err:
                return err
            feat_hash = model_info['strip_hash'](record, feat12)
            rec = MEMORY.get_diag(feat_hash)
            if not rec:
                return _err(
                    "NO_HISTORY",
                    "未找到该带钢的历史诊断记录。",
                    recoverable=True,
                    recommended_action="该卷可能尚未诊断过：如需诊断可调用 diagnose_strip_tool。",
                )
            return _ok(
                stripno=rec.get("stripno", ""),
                fault_cn=rec.get("fault_cn"),
                fault_desc=rec.get("fault_desc"),
                confidence=rec.get("confidence"),
                needs_review=rec.get("needs_review"),
                features_named={
                    name: rec.get("features", [])[j]
                    for j, name in enumerate(FEAT12)
                    if j < len(rec.get("features", []))
                },
                updated_at=rec.get("updated_at"),
            )
        except Exception as e:  # noqa: BLE001
            return _err(
                "HISTORY_QUERY_FAILED",
                f"历史诊断查询失败: {e}",
                recoverable=False,
                recommended_action="请告知用户历史查询暂不可用。",
            )

    # 动态补充描述（必需列/故障类型清单建工具时才可得）
    diagnose_strip_tool.description += f"\n必需字段（缺一不可）: {required_desc}"
    search_similar_cases_tool.description += f"\n必需字段: {required_desc}"
    query_hist_diag_tool.description += f"\n必需字段: {required_desc}"
    get_feature_importance_tool.description += f"\n已知故障类型（枚举）: {known_faults_desc}"

    return [
        diagnose_strip_tool,
        retrieve_knowledge_tool,
        search_similar_cases_tool,
        get_feature_importance_tool,
        batch_csv_diagnosis_tool,
        query_hist_diag_tool,
    ]
