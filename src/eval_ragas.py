#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAGAS 评估脚本
--------------
读取 data/eval_dataset.json，计算 Faithfulness, Answer Relevancy,
Context Precision, Context Recall 四项指标，并保存结果至 ragas_eval.csv。
"""

import os
import json
import pandas as pd
from datasets import Dataset

# 环境变量设置（与 app.py 保持一致）
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "你的API-KEY")
os.environ["DEEPSEEK_API_KEY"] = DEEPSEEK_API_KEY

# 项目根目录定位
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent  # 假设 eval_ragas.py 在 src/ 下
MODEL_DIR = str(PROJECT_ROOT / "models")
EMBED_MODEL_PATH = str(PROJECT_ROOT / "models" / "bge-large-zh-v1.5")

# ---------- 1. 初始化 LLM 与 Embedding ----------
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings

llm = ChatOpenAI(
    model="deepseek-v4-flash",
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com",
    temperature=0.1,
)

embeddings = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL_PATH,
    model_kwargs={'device': 'cpu'},  # 如有 GPU 可改为 'cuda'
    encode_kwargs={'normalize_embeddings': True}
)

# ---------- 2. 准备 RAGAS 评估器 ----------
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

ragas_llm = LangchainLLMWrapper(llm)
ragas_embeddings = LangchainEmbeddingsWrapper(embeddings)

# ---------- 3. 加载评估数据集 ----------
eval_path = PROJECT_ROOT / "data" / "eval_dataset.json"
with open(eval_path, "r", encoding="utf-8") as f:
    eval_data = json.load(f)

# 转换为 HuggingFace Dataset 格式
# RAGAS 要求字段: question, answer, contexts, ground_truth
dataset = Dataset.from_list(eval_data)

# ---------- 4. 运行评估 ----------
print(f"正在评估 {len(dataset)} 条数据...")
result = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    llm=ragas_llm,
    embeddings=ragas_embeddings,
)

# ---------- 5. 保存结果 ----------
df = result.to_pandas()
output_path = PROJECT_ROOT / "data" / "ragas_eval.csv"
df.to_csv(output_path, index=False, encoding="utf-8-sig")

print("评估完成！结果摘要：")
print(df[["faithfulness", "answer_relevancy", "context_precision", "context_recall"]].describe())
print(f"详细结果已保存至：{output_path}")