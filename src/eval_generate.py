#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自动生成 RAGAS 评估所需的 contexts 和 answer
输入：data/eval_questions.json (含 question 与 ground_truth)
输出：data/eval_dataset.json (含 question, ground_truth, contexts, answer)
"""

import json
import sys
from pathlib import Path

# 将 src 目录加入 Python 路径，以便导入 app 中的函数
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

# 从 app.py 导入必要的函数和全局资源
# 注意：这会触发 app.py 中初始化资源的代码，因此可能需要较长时间
from app import (
    retrieve_context,
    generate_answer,
    retriever,  # 全局检索器
    feature_names,  # 特征名列表
    model_info,  # 可选，仅用于获取 feature_names
)


def build_eval_dataset(input_path: str, output_path: str):
    """读取问题文件，调用系统生成上下文和答案，保存评估数据集"""

    # 读取输入
    with open(input_path, "r", encoding="utf-8") as f:
        questions_data = json.load(f)

    eval_dataset = []

    for idx, item in enumerate(questions_data):
        question = item["question"]
        ground_truth = item.get("ground_truth", "")

        print(f"处理 [{idx + 1}/{len(questions_data)}]: {question[:50]}...")

        try:
            # 1. 检索上下文 (纯知识问答模式，无特征向量)
            context, sim_text = retrieve_context(
                retriever,
                query=question,
                feat_vec=None,
                fault_cn=None
            )
            # 如果相似案例文本不为空，可考虑拼接到 context 中
            # 这里为了简单，仅使用 context，也可选择同时使用
            if sim_text:
                context = context + "\n" + sim_text

            # 2. 生成回答 (diagnosis 参数传 None 表示纯问答)
            answer = generate_answer(
                diagnosis=None,
                context=context,
                sim_text=sim_text if sim_text else "",
                user_question=question,
                feature_names=feature_names
            )

            # 如果答案为空或包含错误，标记为无效
            if not answer or "生成失败" in answer or "API 密钥未配置" in answer:
                print(f"  警告: 回答生成失败，跳过该条")
                continue

            eval_dataset.append({
                "question": question,
                "ground_truth": ground_truth,
                "contexts": [context],  # RAGAS 要求 contexts 为列表
                "answer": answer
            })

        except Exception as e:
            print(f"  错误: {e}")
            continue

    # 保存结果
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(eval_dataset, f, ensure_ascii=False, indent=2)

    print(f"\n评估数据集生成完成，共 {len(eval_dataset)} 条，保存至 {output_path}")


if __name__ == "__main__":
    import os

    # 设置 API key 环境变量（若未设置）
    if "DEEPSEEK_API_KEY" not in os.environ:
        os.environ["DEEPSEEK_API_KEY"] = "your-api-key"  # 替换为实际 key

    build_eval_dataset(
        input_path="data/eval_questions.json",
        output_path="data/eval_dataset.json"
    )