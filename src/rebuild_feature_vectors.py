#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
仅重建 fault_feature_vectors 集合（ckpt 坐标系版）。
- 特征列序与标准化统计量取自 cnn_hybrid_diag.pt（FEAT12 业务序 + feat_mean/std），
  与查询端 scale_feat12 同一坐标系；不得使用 model_config.pkl 的旧 scaler
  （MICAN 时代字母序，与 FEAT12 逐维错位，会导致相似度失真）。
- 不影响 rag_knowledge 文本集合，无需重新嵌入文档，速度很快。
- 用法: python src/rebuild_feature_vectors.py
"""
import json
from pathlib import Path
from pymilvus import MilvusClient, DataType

from hybrid_model import FEAT12, CkptScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
QUADRUPLE_JSON = str(PROJECT_ROOT / "data" / "quadruplets" / "width.json")
MILVUS_DB_PATH = str(PROJECT_ROOT / "milvus_kb.db")
COLLECTION_FEATURES = "fault_feature_vectors"

# 四元组 JSON / 原始 CSV 的列序（与 test_samples.csv 表头一致）
QUAD_FEATURE_ORDER = [
    "is_FM", "is_RM", "is_DM", "is_HT", "is_WS", "is_NA",
    "FMWTARGETHOT", "RDWTARGETTOTAL", "PDIWIDTHTOL",
    "FMWIDTHACTHOT", "RMWIDTHACTHOT", "FMWTARGETCOL",
]


def reorder_features_to_model_order(raw_vector, feature_names):
    """将四元组/CSV 列序的 12 维特征重排为模型 feature_names 列序。

    注意：本函数是 kb_utils.reorder_features_to_model_order 的副本（此处不 import
    kb_utils 是为了避开 langchain 依赖）。两份实现若需修改必须同步，否则入库端与
    查询端会落到不同坐标系，相似案例的余弦值将失去意义。

    两份副本同语义：列名不在 QUAD_FEATURE_ORDER 内时抛 ValueError，
    让列序漂移在入库前响亮失败（错序向量一旦被标准化入库，相似度会全面
    失真且没有任何报错）。
    """
    if len(raw_vector) != 12 or len(feature_names) != 12:
        return raw_vector
    pos = [QUAD_FEATURE_ORDER.index(name) for name in feature_names]
    return [raw_vector[i] for i in pos]


def main():
    # 停服提示置顶（原在 main 末尾——重建跑完才提示已无操作意义）：
    # drop+create 与运行中的服务并发读写 milvus_kb.db 可能锁冲突
    print("⚠️  停服提示：本脚本将 drop 并重建故障特征向量集合。若主服务或")
    print("    知识库在线更新页正在运行，请先停止它们再执行本脚本。")
    print("构造入库端标准化器（ckpt 统计量，FEAT12 业务序）...")
    scaler = CkptScaler(FEAT12)
    feature_names = FEAT12

    print(f"读取四元组: {QUADRUPLE_JSON}")
    with open(QUADRUPLE_JSON, encoding="utf-8") as f:
        quads = json.load(f)

    vectors, labels, plans = [], [], []
    skipped = 0
    for q in quads:
        try:
            v = [float(x) for x in q["故障特征"].replace("，", ",").split(",")]
            if len(v) != 12:
                skipped += 1
                continue
            # 列序对不上时本函数抛 ValueError（不再静默返回原序向量），
            # 单独捕获并明确报错，避免"错序向量静默入库"这种无痕失真
            v = reorder_features_to_model_order(v, feature_names)
            vectors.append(scaler.transform([v])[0].tolist())
            labels.append(q["故障类型"])
            plans.append(q["诊断方案"])
        except ValueError as e:
            print(f"  [错误] 特征列序与 FEAT12 不匹配，已中止：{e}")
            print(f"  请核对 QUAD_FEATURE_ORDER / hybrid_model.FEAT12 与四元组特征字符串")
            return
        except Exception as e:  # noqa: BLE001
            skipped += 1
            print(f"  [跳过] 四元组解析失败: {e}")
            continue
    print(f"有效四元组特征向量: {len(vectors)} 条（跳过 {skipped} 条）")
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

    # 相似案例缓存与文本检索缓存均以本集合为数据源，重建后必须失效，
    # 否则运行中的服务会继续返回旧集合的相似度与诊断方案。
    # 说明：本脚本不 import service，因此本进程没有注册任何进程内失效回调——
    # 这里实际清理的是持久层缓存（Redis 键 / 文件缓存）。对**运行中的**主服务，
    # 本进程无法直接清它的内存缓存，依赖服务侧按集合目录时间戳自动感知。
    try:
        from memory_store import get_memory_store
        n = get_memory_store().invalidate_retrieve_cache()
        print(f"已失效持久层检索缓存 {n} 条（运行中的服务将按集合时间戳自动感知）")
    except Exception as e:  # noqa: BLE001
        print(f"检索缓存失效失败（重启服务后生效）: {e}")


if __name__ == "__main__":
    main()
