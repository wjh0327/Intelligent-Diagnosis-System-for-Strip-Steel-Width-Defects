#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAGAS 评估脚本（ragas 0.4.x，已在 steel 环境 ragas==0.4.3 上验证 API）
--------------------------------------------------------------------
读取 data/eval_dataset.json（由 eval_generate.py 生成，字段：
user_input / reference / retrieved_contexts / response），
计算 faithfulness、answer_relevancy、context_precision、context_recall，
结果保存至 data/ragas_eval.csv。

用法（项目根目录执行）：python src/eval_ragas.py
依赖：先运行 python src/eval_generate.py 生成评估数据集。
"""

import json
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
sys.path.insert(0, str(SRC_DIR))  # service 等模块按平级目录导入

from agent_graph import LLM_MAX_RETRIES, LLM_TIMEOUT  # noqa: E402
from service import DEEPSEEK_API_KEY, EMBED_MODEL_PATH, LLM_MODEL  # noqa: E402

from langchain_huggingface import HuggingFaceEmbeddings  # noqa: E402
from langchain_openai import ChatOpenAI  # noqa: E402
from ragas import RunConfig, evaluate  # noqa: E402
from ragas.dataset_schema import EvaluationDataset  # noqa: E402
from ragas.embeddings import LangchainEmbeddingsWrapper  # noqa: E402
from ragas.llms import LangchainLLMWrapper  # noqa: E402
from ragas.metrics import (  # noqa: E402
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

METRIC_COLS = [
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
]


def main() -> None:
    eval_path = PROJECT_ROOT / "data" / "eval_dataset.json"
    if not eval_path.exists():
        sys.exit(
            f"评估数据集不存在: {eval_path}\n请先运行 python src/eval_generate.py 生成。"
        )

    # 评估 LLM：复用 agent_graph 的超时/重试环境变量配置
    llm = ChatOpenAI(
        model=LLM_MODEL,
        api_key=DEEPSEEK_API_KEY,
        base_url="https://api.deepseek.com",
        temperature=0.1,
        timeout=LLM_TIMEOUT,
        max_retries=LLM_MAX_RETRIES,
    )
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:  # pragma: no cover
        device = "cpu"
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL_PATH,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )

    ragas_llm = LangchainLLMWrapper(llm)
    ragas_embeddings = LangchainEmbeddingsWrapper(embeddings)
    # 注意：RunConfig.timeout 是"单样本单指标"整个作业的总时限（faithfulness 需多次
    # 串行 LLM 调用，60s 会整体超时），而非单次 HTTP 调用超时，须远大于 LLM_TIMEOUT
    run_config = RunConfig(timeout=300, max_retries=LLM_MAX_RETRIES)
    # DeepSeek API 仅支持 n=1，answer_relevancy 默认 strictness=3（并行生成 3 个
    # 问题变体）会触发 400 Invalid n value，降为 1
    answer_relevancy.strictness = 1

    eval_data = json.loads(eval_path.read_text(encoding="utf-8"))
    if not eval_data:
        sys.exit("评估数据集为空，请检查 eval_generate.py 的生成结果。")
    dataset = EvaluationDataset.from_list(eval_data)

    print(f"正在评估 {len(dataset)} 条数据...")
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=ragas_llm,
        embeddings=ragas_embeddings,
        run_config=run_config,
        show_progress=True,
    )

    df = result.to_pandas()
    output_path = PROJECT_ROOT / "data" / "ragas_eval.csv"
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print("评估完成！结果摘要：")
    print(df[METRIC_COLS].describe())
    print(f"详细结果已保存至：{output_path}")


if __name__ == "__main__":
    if not DEEPSEEK_API_KEY:
        sys.exit("未设置 DEEPSEEK_API_KEY，无法运行 RAGAS 评估。")
    main()
