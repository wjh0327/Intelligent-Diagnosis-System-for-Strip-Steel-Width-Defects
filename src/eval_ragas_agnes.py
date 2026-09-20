#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAGAS 评估脚本 —— AGNES 网关版
==================================
与 eval_ragas.py 逻辑一致，评估 LLM 改走 AGNES 网关（apihub.agnes-ai.com/v1，
模型 agnes-2.0-flash），后续 ragas 评估的迭代调整在本文件内进行
（eval_ragas.py 保留为 DeepSeek 基线版）。

覆盖环境变量（均可选）：
  AGNES_API_KEY   必需
  AGNES_BASE_URL  默认 https://apihub.agnes-ai.com/v1
  AGNES_MODEL     默认 agnes-2.0-flash（带 reasoning 输出的模型）

用法（项目根目录执行）：python src/eval_ragas_agnes.py
依赖：先运行 python src/eval_generate.py 生成 data/eval_dataset.json。
"""

import json
import os
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
sys.path.insert(0, str(SRC_DIR))  # agent_graph 等模块按平级目录导入

from agent_graph import LLM_MAX_RETRIES, LLM_TIMEOUT  # noqa: E402
from service import EMBED_MODEL_PATH  # noqa: E402

AGNES_API_KEY = os.getenv("AGNES_API_KEY", "").strip()
AGNES_BASE_URL = os.getenv("AGNES_BASE_URL", "https://apihub.agnes-ai.com/v1").strip()
AGNES_MODEL = os.getenv("AGNES_MODEL", "agnes-2.0-flash").strip()

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

    # 评估 LLM：超时/重试复用 agent_graph 的环境变量配置
    llm = ChatOpenAI(
        model=AGNES_MODEL,
        api_key=AGNES_API_KEY,
        base_url=AGNES_BASE_URL,
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
    # 串行 LLM 调用，60s 会整体超时），而非单次 HTTP 调用超时，须远大于 LLM_TIMEOUT。
    # 探针实测：faithfulness 判定阶段要对长回答的全部 claims 生成逐条判定，
    # 生成该结构化输出可达 5 分钟以上，300s 仍有样本超时，故放宽到 900s；
    # 作业默认 16 路并行，墙钟时间≈最慢作业
    run_config = RunConfig(timeout=900, max_retries=LLM_MAX_RETRIES,
                           max_workers=2)  # AGNES 免费档有严格速率限制，16 路并发会触发 429，
                                           # 降为 2 路串行推进；墙钟时间换完整性
    # 网关模型普遍仅支持 n=1，answer_relevancy 默认 strictness=3（并行生成 3 个
    # 问题变体）会触发 400 Invalid n value，降为 1
    answer_relevancy.strictness = 1

    eval_data = json.loads(eval_path.read_text(encoding="utf-8"))
    if not eval_data:
        sys.exit("评估数据集为空，请检查 eval_generate.py 的生成结果。")
    dataset = EvaluationDataset.from_list(eval_data)

    print(f"正在评估 {len(dataset)} 条数据（LLM: {AGNES_MODEL} @ {AGNES_BASE_URL}）...")
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=ragas_llm,
        embeddings=ragas_embeddings,
        run_config=run_config,
        show_progress=True,
    )

    df = result.to_pandas()
    output_path = PROJECT_ROOT / "data" / "ragas_eval_agnes.csv"
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print("评估完成！结果摘要：")
    print(df[METRIC_COLS].describe())
    print(f"详细结果已保存至：{output_path}")


if __name__ == "__main__":
    if not AGNES_API_KEY:
        sys.exit("未设置 AGNES_API_KEY，无法运行 RAGAS 评估。")
    main()
