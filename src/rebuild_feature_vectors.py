#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
仅重建 fault_feature_vectors 集合（修复四元组特征列序 bug 后的特征向量）。
- 不影响 rag_knowledge 文本集合，无需重新嵌入文档，速度很快。
- 用法: python src/rebuild_feature_vectors.py
"""
import json
from pathlib import Path
import joblib
from pymilvus import MilvusClient, DataType

PROJECT_ROOT = Path(__file__).resolve().parent.parent
QUADRUPLE_JSON = str(PROJECT_ROOT / "data" / "quadruplets" / "width.json")
MILVUS_DB_PATH = r"F:\RAG Agent\milvus_kb.db"
MODEL_CONFIG_PATH = str(PROJECT_ROOT / "models" / "model_config.pkl")
COLLECTION_FEATURES = "fault_feature_vectors"

# 四元组 JSON / 原始 CSV 的列序（与 test_samples.csv 表头一致）
QUAD_FEATURE_ORDER = [
    "is_FM", "is_RM", "is_DM", "is_HT", "is_WS", "is_NA",
    "FMWTARGETHOT", "RDWTARGETTOTAL", "PDIWIDTHTOL",
    "FMWIDTHACTHOT", "RMWIDTHACTHOT", "FMWTARGETCOL",
]


def reorder_features_to_model_order(raw_vector, feature_names):
    """将四元组/CSV 列序的 12 维特征重排为模型 feature_names 列序"""
    if len(raw_vector) != 12 or len(feature_names) != 12:
        return raw_vector
    try:
        pos = [QUAD_FEATURE_ORDER.index(name) for name in feature_names]
        return [raw_vector[i] for i in pos]
    except ValueError:
        return raw_vector


def main():
    print("加载模型配置...")
    config = joblib.load(MODEL_CONFIG_PATH)
    scaler = config["scaler"]
    feature_names = config["feature_names"]

    print(f"读取四元组: {QUADRUPLE_JSON}")
    with open(QUADRUPLE_JSON, encoding="utf-8") as f:
        quads = json.load(f)

    vectors, labels, plans = [], [], []
    for q in quads:
        try:
            v = [float(x) for x in q["故障特征"].replace("，", ",").split(",")]
            if len(v) != 12:
                continue
            v = reorder_features_to_model_order(v, feature_names)
            vectors.append(scaler.transform([v])[0].tolist())
            labels.append(q["故障类型"])
            plans.append(q["诊断方案"])
        except Exception:
            continue
    print(f"有效四元组特征向量: {len(vectors)} 条")
    if not vectors:
        print("无有效向量，退出")
        return

    print(f"连接 Milvus Lite: {MILVUS_DB_PATH}")
    client = MilvusClient(MILVUS_DB_PATH)

    if client.has_collection(COLLECTION_FEATURES):
        client.drop_collection(COLLECTION_FEATURES)
        print(f"已删除旧集合 {COLLECTION_FEATURES}")

    schema = client.create_schema(auto_id=True)
    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field("vector", DataType.FLOAT_VECTOR, dim=12)
    schema.add_field("label", DataType.VARCHAR, max_length=100)
    schema.add_field("plan", DataType.VARCHAR, max_length=4000)

    index = client.prepare_index_params()
    index.add_index("vector", index_type="IVF_FLAT",
                    metric_type="COSINE", params={"nlist": 64})
    client.create_collection(COLLECTION_FEATURES, schema=schema, index_params=index)
    print(f"集合 {COLLECTION_FEATURES} 创建成功")

    data = [
        {"vector": vec, "label": lab, "plan": plan}
        for vec, lab, plan in zip(vectors, labels, plans)
    ]
    client.insert(COLLECTION_FEATURES, data)
    try:
        client.flush(COLLECTION_FEATURES)
        print("已 flush，manifest 已更新")
    except Exception as e:
        print(f"flush 失败（不影响数据，WAL 可恢复）: {e}")
    client.close()
    print(f"重建完成: 已插入 {len(data)} 条标准化故障特征向量（模型列序）")


if __name__ == "__main__":
    main()
