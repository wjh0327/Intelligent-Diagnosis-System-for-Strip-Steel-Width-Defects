#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自动生成 RAGAS 评估数据集（评估当前系统真实行为）
输入：data/eval_questions.json   —— [{question, ground_truth}, ...]
输出：data/eval_dataset.json     —— ragas 0.4 schema：
      [{user_input, reference, retrieved_contexts, response}, ...]

与前版的区别：不再依赖旧版 app.py 的 generate_answer（已删除），
改走 service.init_resources() 构建的完整 Agent 链路：
  - response          取自 agent_graph.invoke 的 final_answer
  - retrieved_contexts 取自推理过程中检索类工具（retrieve_knowledge_tool /
    search_similar_cases_tool）的实际返回；Agent 未检索时回退为
    直接调用 retrieve_context，保证 contexts 不为空列表

用法（项目根目录执行）：python src/eval_generate.py
"""

import json
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
sys.path.insert(0, str(SRC_DIR))  # service 等模块按平级目录导入

from logger import get_logger  # noqa: E402
from service import DEEPSEEK_API_KEY, init_resources, retrieve_context  # noqa: E402

logger = get_logger("eval_generate")

# 其返回内容属于"检索到的知识"的工具（计入 retrieved_contexts）
RETRIEVAL_TOOLS = {"retrieve_knowledge_tool", "search_similar_cases_tool"}


def _extract_knowledge(obs: str) -> str:
    """从检索工具的返回中提取知识文本。

    新版工具返回 JSON（success/knowledge 或 cases）；success 为 false 的空结果
    返回空串。回退兼容旧的纯文本与 "[检索提示]" 格式。
    """
    if not obs:
        return ""
    try:
        obj = json.loads(obs)
        if isinstance(obj, dict):
            if obj.get("success") is False:
                return ""
            return str(obj.get("knowledge") or obj.get("cases") or "")
    except Exception:  # noqa: BLE001
        pass
    return "" if obs.startswith("[检索提示]") else obs


def build_eval_dataset(input_path: Path, output_path: Path) -> None:
    """逐题调用 Agent 生成 contexts 与 answer，保存 ragas 评估数据集。"""
    model_info, retriever, feature_names, agent_graph = init_resources()

    questions_data = json.loads(input_path.read_text(encoding="utf-8"))
    eval_dataset = []

    for idx, item in enumerate(questions_data):
        question = item["question"]
        ground_truth = item.get("ground_truth", "")
        print(f"处理 [{idx + 1}/{len(questions_data)}]: {question[:50]}...")

        try:
            # 纯知识问答：无特征、无 CSV、无历史，逐题独立评估
            initial_state = {
                "query": question,
                "features": "",
                "csv_path": "",
                "conversation_history": [],
                "intermediate_steps": [],
                "final_answer": "",
                "step_count": 0,
                "batch_done": False,
                "batch_summary": "",
            }
            final_state = agent_graph.invoke(initial_state)

            answer = (final_state.get("final_answer") or "").strip()
            if not answer or "LLM 调用失败" in answer:
                print("  警告: Agent 未生成有效回答，跳过该条")
                continue

            # 取 Agent 推理中检索工具的真实返回作为 contexts
            contexts = []
            for name, obs in final_state.get("intermediate_steps", []):
                if name not in RETRIEVAL_TOOLS:
                    continue
                knowledge = _extract_knowledge(obs)
                if knowledge:
                    contexts.append(knowledge.strip())
            if not contexts:
                context, _ = retrieve_context(
                    retriever, query=question, feat_vec=None, fault_cn=None
                )
                contexts = [context] if context else []

            eval_dataset.append({
                "user_input": question,
                "reference": ground_truth,
                "retrieved_contexts": contexts,
                "response": answer,
            })

        except Exception as e:  # noqa: BLE001
            print(f"  错误: {e}")
            continue

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(eval_dataset, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n评估数据集生成完成，共 {len(eval_dataset)} 条，保存至 {output_path}")


if __name__ == "__main__":
    if not DEEPSEEK_API_KEY:
        sys.exit("未设置 DEEPSEEK_API_KEY，无法生成评估数据（Agent 问答依赖 LLM）。")
    build_eval_dataset(
        input_path=PROJECT_ROOT / "data" / "eval_questions.json",
        output_path=PROJECT_ROOT / "data" / "eval_dataset.json",
    )
