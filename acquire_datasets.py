#!/usr/bin/env python3
"""
Stage S1 — Dataset Acquisition & Integrity

从 Hugging Face 下载软件修复训练数据集，导出为统一 JSONL，并做基础完整性审计。

支持的数据源（默认 HF ID，可通过 --dataset-id 覆盖）：
  - swe_fixer:   SWE-Fixer-Train-110K (真实 Issue-Patch, ~115K)
  - swe_smith_py: SWE-smith-py (Python 修复任务, 有 FAIL_TO_PASS/PASS_TO_PASS)

输出：
  <output_dir>/raw/<source>.jsonl          标准化后的原始记录
  <output_dir>/raw/<source>.provenance.json 来源与审计元信息

用法（本地 Smoke Test，不下载全量）：
  python src/acquire_datasets.py --sources swe_fixer --max-rows 200 --dry-run

用法（正式采集）：
  python src/acquire_datasets.py --sources swe_fixer swe_smith_py
"""

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    from datasets import load_dataset
    from huggingface_hub import HfApi
except Exception as e:  # pragma: no cover - 依赖缺失时给出清晰错误
    print(f"FATAL: 缺少依赖: {e}. 请先 pip install datasets huggingface_hub", file=sys.stderr)
    sys.exit(3)

# 默认数据集映射（HF dataset_id, 是否已知可用）
DEFAULT_DATASETS = {
    "swe_fixer": "princeton-nlp/SWE-Fixer-Train-110K",
    "swe_smith_py": "swe-smith/SWE-smith-py",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize_record(src: str, ds_id: str, idx: int, rec: dict) -> dict | None:
    """把原始数据集记录归一化为统一 Schema。

    不同数据集字段名不一致，这里做尽力映射。不可解析的记录返回 None。
    归一化后保留原始字段在 _raw 中，方便审计。
    """
    if not isinstance(rec, dict):
        return None

    # 尽力提取通用字段（不同数据集命名不同，取第一个存在的）
    def first(*keys):
        for k in keys:
            if k in rec and rec[k] is not None:
                return rec[k]
        return None

    instance_id = first("instance_id", "id", "task_instance_id", "sample_id")
    problem = first("problem_statement", "issue", "description", "prompt", "text")
    patch = first("patch", "gold_patch", "diff", "solution", "answer")
    test_patch = first("test_patch", "test_diff", "test_patch_code")
    fail_to_pass = first("fail_to_pass", "fail_to_pass_tests")
    pass_to_pass = first("pass_to_pass", "pass_to_pass_tests")
    repo = first("repo", "repository", "project")
    base_commit = first("base_commit", "base_commit_sha", "commit", "parent_commit")

    # 最小可用性检查：必须有 instance_id 和 问题描述
    if not instance_id or not problem:
        return None

    normalized = {
        "source": src,
        "dataset_id": ds_id,
        "idx": idx,
        "instance_id": str(instance_id),
        "problem_statement": str(problem),
        "patch": str(patch) if patch is not None else "",
        "test_patch": str(test_patch) if test_patch is not None else "",
        "repo": str(repo) if repo else "",
        "base_commit": str(base_commit) if base_commit else "",
        "fail_to_pass": fail_to_pass if isinstance(fail_to_pass, (list, str)) else [],
        "pass_to_pass": pass_to_pass if isinstance(pass_to_pass, (list, str)) else [],
        "_raw": rec,
    }

    # provenance 字段（标签来源追踪的前置）
    normalized["_provenance"] = {
        "label_source": "dataset",        # 目前来自数据集，后续可升级为 human/model
        "label_strength": "strong" if patch else "weak",
        "verified": False,                # 尚未经过自动/人工验证，需后续 Gate
        "confidence": 1.0 if patch else 0.0,
    }

    return normalized


def basic_integrity_check(records: list[dict]) -> dict:
    """基础完整性检查，返回统计，不修改记录。"""
    total = len(records)
    has_problem = sum(1 for r in records if r.get("problem_statement"))
    has_patch = sum(1 for r in records if r.get("patch"))
    has_instance_id = sum(1 for r in records if r.get("instance_id"))
    empty_patch = sum(1 for r in records if not r.get("patch"))
    unique_ids = len(set(r.get("instance_id", "") for r in records if r.get("instance_id")))
    return {
        "total_records": total,
        "has_problem_statement": has_problem,
        "has_patch": has_patch,
        "has_instance_id": has_instance_id,
        "empty_patch": empty_patch,
        "unique_instance_ids": unique_ids,
        "duplicate_rate": round(1 - unique_ids / max(total, 1), 4),
    }


def acquire_source(source: str, ds_id: str, max_rows: int | None, out_dir: Path,
                   dry_run: bool) -> dict:
    """下载并导出单个数据源。"""
    result = {
        "source": source,
        "dataset_id": ds_id,
        "status": "started",
        "error": None,
        "rows_downloaded": 0,
        "rows_written": 0,
        "output_file": None,
        "provenance_file": None,
        "integrity": None,
        "sha256": None,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / f"{source}.jsonl"
    prov_path = out_dir / f"{source}.provenance.json"

    if dry_run:
        result["status"] = "dry_run_ok"
        result["rows_downloaded"] = 0
        result["rows_written"] = 0
        result["output_file"] = str(jsonl_path)
        result["provenance_file"] = str(prov_path)
        print(f"[DRY-RUN] {source}: would load {ds_id}, max_rows={max_rows}")
        return result

    print(f"[ACQUIRE] {source}: loading {ds_id} ...")
    try:
        ds = load_dataset(ds_id, split="train")
    except Exception as e:
        # 尝试不指定 split
        try:
            ds = load_dataset(ds_id)
            # 取第一个 split
            split_names = list(ds.keys())
            ds = ds[split_names[0]]
            print(f"[ACQUIRE] {source}: used split '{split_names[0]}'")
        except Exception as e2:
            result["status"] = "failed"
            result["error"] = f"load_dataset failed: {e}; fallback failed: {e2}"
            print(f"[ERROR] {source}: {result['error']}", file=sys.stderr)
            return result

    total_available = len(ds)
    limit = total_available if max_rows is None else min(max_rows, total_available)
    print(f"[ACQUIRE] {source}: total_available={total_available}, limit={limit}")

    written = 0
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for i in range(limit):
            rec = ds[i]
            norm = normalize_record(source, ds_id, i, rec)
            if norm is None:
                continue
            f.write(json.dumps(norm, ensure_ascii=False) + "\n")
            written += 1

    result["rows_downloaded"] = limit
    result["rows_written"] = written
    result["output_file"] = str(jsonl_path)
    result["sha256"] = sha256_file(jsonl_path)

    # 读回做完整性检查
    records = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    result["integrity"] = basic_integrity_check(records)

    # 写 provenance
    provenance = {
        "source": source,
        "dataset_id": ds_id,
        "download_time_utc": datetime.now(timezone.utc).isoformat(),
        "huggingface_dataset_id": ds_id,
        "rows_available": total_available,
        "rows_requested": max_rows,
        "rows_written": written,
        "output_file": str(jsonl_path),
        "sha256": result["sha256"],
        "integrity": result["integrity"],
        "note": "Raw export from HF. Label provenance to be enriched in later stages.",
    }
    prov_path.write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
    result["provenance_file"] = str(prov_path)
    result["status"] = "completed"

    print(f"[DONE] {source}: wrote {written} records to {jsonl_path}")
    print(f"       integrity: {json.dumps(result['integrity'], ensure_ascii=False)}")
    return result


def main():
    ap = argparse.ArgumentParser(description="Stage S1: Dataset Acquisition & Integrity")
    ap.add_argument("--sources", nargs="+", default=["swe_fixer"],
                    choices=list(DEFAULT_DATASETS.keys()),
                    help="要采集的数据源")
    ap.add_argument("--dataset-id", nargs="+", default=None,
                    help="可选：覆盖默认 HF dataset ID，顺序需与 --sources 一一对应")
    ap.add_argument("--max-rows", type=int, default=None,
                    help="每个源最多采集行数（Smoke Test 建议 200）")
    ap.add_argument("--output-dir", default=r"F:\test\gemma4\training\data\raw",
                    help="原始数据输出目录")
    ap.add_argument("--dry-run", action="store_true",
                    help="仅验证环境和参数，不实际下载")
    args = ap.parse_args()

    # 解析 dataset-id 覆盖
    ds_ids = {}
    if args.dataset_id:
        if len(args.dataset_id) != len(args.sources):
            print("FATAL: --dataset-id 数量必须与 --sources 一致", file=sys.stderr)
            sys.exit(2)
        for src, did in zip(args.sources, args.dataset_id):
            ds_ids[src] = did
    else:
        ds_ids = DEFAULT_DATASETS

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for src in args.sources:
        res = acquire_source(src, ds_ids[src], args.max_rows, out_dir, args.dry_run)
        results.append(res)

    # 汇总
    print("\n" + "=" * 60)
    print("S1 Acquisition Summary")
    print("=" * 60)
    all_ok = True
    for r in results:
        status = r["status"]
        if status != "completed" and status != "dry_run_ok":
            all_ok = False
        print(f"  {r['source']}: {status} | rows={r['rows_written']} | {r['output_file']}")
        if r["error"]:
            print(f"    error: {r['error']}")

    summary_path = out_dir / "acquisition_summary.json"
    summary_path.write_text(
        json.dumps({"summary": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSummary saved to {summary_path}")

    if not all_ok and not args.dry_run:
        print("\nWARNING: 部分数据源采集失败，请检查错误信息后再继续。")
        sys.exit(1)


if __name__ == "__main__":
    main()
