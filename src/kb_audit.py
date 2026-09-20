#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/kb_audit.py —— 知识库体检（只读，不修改任何数据/库）
=========================================================
1. 逐 PDF 审计：用与构建管线相同的解析器（kb_utils.load_pdf_sorted_by_columns），
   统计碎行率、平均行长、全角混用率、页均字符数，输出问题文档清单；
   无文本层的扫描件（解析页数 << 物理页数）单独标记。
2. 审计 Milvus rag_knowledge 集合中已存入的文本块：
   统计垃圾块数量、按来源文档聚合，给出污染最重的文档排名。

判定阈值（保守，供人工复核，不是硬标准）：
  垃圾候选: 碎行率 > 0.5，或平均行长 < 4 且字符数 > 50
  可疑:     碎行率 > 0.3，或全角混用率 > 0.15
  其余:     健康

用法（项目根目录执行）: python src/kb_audit.py
输出: 控制台报告 + data/kb_audit_report.csv
"""

import csv
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

import fitz  # PyMuPDF

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
sys.path.insert(0, str(SRC_DIR))

from kb_utils import load_pdf_sorted_by_columns  # noqa: E402
from logger import get_logger  # noqa: E402

logger = get_logger("kb_audit")

DOCUMENTS_DIR = PROJECT_ROOT / "data" / "documents"
MILVUS_DB_PATH = PROJECT_ROOT / "milvus_kb.db"
REPORT_PATH = PROJECT_ROOT / "data" / "kb_audit_report.csv"

# 全角 ASCII 区块 U+FF01–U+FF5E（字母数字标点全算），比旧过滤器只认字母数字更严
FW_RE = re.compile(r"[\uff01-\uff5e]")
CN_RE = re.compile(r"[\u4e00-\u9fff]")
ALNUM_RE = re.compile(r"[A-Za-z0-9]")


def text_metrics(text: str) -> dict:
    """对一段文本计算质量指标。"""
    lines = [l.strip() for l in (text or "").splitlines() if l.strip()]
    n_lines = len(lines)
    if n_lines == 0:
        return {"lines": 0, "chars": 0, "frag": 0.0, "avg_len": 0.0, "fw_ratio": 0.0}
    core_lens = [len(re.sub(r"\s", "", l)) for l in lines]
    frag = sum(1 for n in core_lens if n < 3) / n_lines
    avg_len = sum(core_lens) / n_lines
    fw = len(FW_RE.findall(text))
    cn = len(CN_RE.findall(text))
    an = len(ALNUM_RE.findall(text))
    fw_ratio = fw / max(fw + cn + an, 1)
    return {
        "lines": n_lines,
        "chars": sum(core_lens),
        "frag": round(frag, 4),
        "avg_len": round(avg_len, 2),
        "fw_ratio": round(fw_ratio, 4),
    }


def verdict(m: dict) -> str:
    if m["lines"] == 0 or m["chars"] == 0:
        return "空"
    if m["frag"] > 0.5 or (m["avg_len"] < 4 and m["chars"] > 50):
        return "垃圾候选"
    if m["frag"] > 0.3 or m["fw_ratio"] > 0.15:
        return "可疑"
    return "健康"


def audit_pdf(pdf_path: Path) -> dict:
    """单个 PDF：物理页数 vs 解析页数 + 整体质量指标 + 垃圾页计数。"""
    docs = load_pdf_sorted_by_columns(str(pdf_path))
    with fitz.open(pdf_path) as pdf_doc:
        physical_pages = len(pdf_doc)

    page_metrics = [text_metrics(d.page_content) for d in docs]
    all_text = "\n".join(d.page_content for d in docs)
    overall = text_metrics(all_text)
    bad_pages = sum(1 for m in page_metrics if verdict(m) == "垃圾候选")

    if not docs and physical_pages > 0:
        v = "无文本层扫描件"
    else:
        v = verdict(overall)
    return {
        "file": pdf_path.name,
        "physical_pages": physical_pages,
        "parsed_pages": len(docs),
        "chunks_chars": overall["chars"],
        "frag": overall["frag"],
        "avg_len": overall["avg_len"],
        "fw_ratio": overall["fw_ratio"],
        "bad_pages": bad_pages,
        "verdict": v,
    }


def audit_milvus() -> dict:
    """审计已存入 Milvus 的文本块：按 doc_type 计数，垃圾块按来源文档聚合。"""
    from pymilvus import MilvusClient

    client = MilvusClient(str(MILVUS_DB_PATH))
    client.load_collection("rag_knowledge")
    rows = client.query(
        collection_name="rag_knowledge",
        filter='doc_type in ["document", "quadruple_text", "quadruple_kw"]',
        output_fields=["text", "doc_type", "metadata"],
        limit=16384,
    )
    client.close()

    type_counts = Counter(r.get("doc_type", "?") for r in rows)
    chunk_stats = []
    for r in rows:
        if r.get("doc_type") != "document":  # 四元组是结构化文本，不参与垃圾判定
            continue
        m = text_metrics(r.get("text", ""))
        meta = r.get("metadata") or {}
        if isinstance(meta, str):
            import json as _json
            try:
                meta = _json.loads(meta)
            except Exception:  # noqa: BLE001
                meta = {}
        v = verdict(m)
        if v in ("垃圾候选", "可疑"):
            chunk_stats.append({
                "source": (meta or {}).get("source", "?"),
                "page": (meta or {}).get("page", "?"),
                "verdict": v,
                **m,
                "preview": r.get("text", "")[:80].replace("\n", "␤"),
            })
    return {"total_rows": len(rows), "type_counts": type_counts,
            "bad_chunks": chunk_stats}


def main() -> None:
    print("=" * 90)
    print("知识库体检（只读）")
    print("=" * 90)

    # ---------- 1. PDF 文档审计 ----------
    pdf_files = sorted(DOCUMENTS_DIR.glob("**/*.pdf"))
    print(f"\n[1/2] PDF 文档审计：{len(pdf_files)} 个文件（使用与构建管线相同的解析器）\n")
    header = f"{'文件名':<52}{'物理页':>4}{'解析页':>4}{'字符数':>8}{'碎行率':>7}{'平均行长':>7}{'全角率':>7}{'垃圾页':>5}  判定"
    print(header)
    print("-" * 90)
    results = []
    for p in pdf_files:
        try:
            r = audit_pdf(p)
        except Exception as e:  # noqa: BLE001
            r = {"file": p.name, "physical_pages": -1, "parsed_pages": -1,
                 "chunks_chars": -1, "frag": -1, "avg_len": -1, "fw_ratio": -1,
                 "bad_pages": -1, "verdict": f"解析失败: {e}"}
        results.append(r)
        print(f"{r['file']:<52}{r['physical_pages']:>4}{r['parsed_pages']:>4}"
              f"{r['chunks_chars']:>8}{r['frag']:>7.2f}{r['avg_len']:>7.1f}"
              f"{r['fw_ratio']:>7.2f}{r['bad_pages']:>5}  {r['verdict']}")

    n_bad = sum(1 for r in results if r["verdict"] in ("垃圾候选", "可疑", "无文本层扫描件"))
    print("-" * 90)
    print(f"文档判定汇总：垃圾候选 {sum(1 for r in results if r['verdict']=='垃圾候选')}"
          f" | 可疑 {sum(1 for r in results if r['verdict']=='可疑')}"
          f" | 无文本层 {sum(1 for r in results if r['verdict']=='无文本层扫描件')}"
          f" | 健康 {sum(1 for r in results if r['verdict']=='健康')}")

    # ---------- 2. Milvus 已存块审计 ----------
    print(f"\n[2/2] Milvus 已存块审计：{MILVUS_DB_PATH.name} / rag_knowledge\n")
    try:
        mv = audit_milvus()
        print(f"总块数 {mv['total_rows']}，按类型: {dict(mv['type_counts'])}")
        bad = mv["bad_chunks"]
        if bad:
            by_source = defaultdict(lambda: [0, 0])
            for c in bad:
                by_source[c["source"]][0] += 1
                if c["verdict"] == "垃圾候选":
                    by_source[c["source"]][1] += 1
            print(f"问题块共 {len(bad)} 个（垃圾候选 {sum(1 for c in bad if c['verdict']=='垃圾候选')}"
                  f" / 可疑 {sum(1 for c in bad if c['verdict']=='可疑')}），按来源文档：\n")
            print(f"{'来源文档':<52}{'垃圾候选':>6}{'可疑':>5}")
            for src, (n_all, n_junk) in sorted(by_source.items(), key=lambda x: -x[1][0]):
                print(f"{src:<52}{n_junk:>6}{n_all - n_junk:>5}")
            print("\n最严重的 3 个问题块示例：")
            for c in sorted(bad, key=lambda x: -x["frag"])[:3]:
                print(f"  [{c['source']} p{c['page']}] 碎行率={c['frag']} 预览: {c['preview'][:60]}")
        else:
            print("未发现垃圾候选/可疑块。")
    except Exception as e:  # noqa: BLE001
        print(f"Milvus 审计失败: {e}")

    # ---------- 3. 报告落盘 ----------
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["file", "physical_pages", "parsed_pages", "chars",
                    "frag_ratio", "avg_line_len", "fw_ratio", "bad_pages", "verdict"])
        for r in results:
            w.writerow([r["file"], r["physical_pages"], r["parsed_pages"],
                        r["chunks_chars"], r["frag"], r["avg_len"],
                        r["fw_ratio"], r["bad_pages"], r["verdict"]])
    print(f"\n报告已保存: {REPORT_PATH}")


if __name__ == "__main__":
    main()
