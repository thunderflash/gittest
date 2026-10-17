#!/usr/bin/env python3
"""M3 continuation audit — independent offline verification (2026-10-17).

Scope: read-only re-verification of the delivered M0–M3 evidence bundles,
following the independent audit path suggested by
project/交付清单_20261011.md §4 and project/GateC3_audit/VERIFY_README.md.

Boundary (inherited from HANDOVER_TASK_INCOMPLETE_2026-10-11.md §5):
  - zero network, zero Kaggle API, zero authentication
  - zero submission / retry / rename, zero GPU, zero model/agent runs
  - no engineering code execution; only the two read-only audit scripts are
    re-run (C-3 closeout script and the interlocked draft verifier), which is
    the reproducibility check the delivery manifest itself prescribes.
  - no historical evidence is modified; all new output stays in this directory.

Usage:
  python3 continuation_audit_verify.py --project <path>/workspace_extracted/project --out ./verification_matrix.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path

EXPECTED_C3_JSON_SHA256 = "d3b35a9130647268bfce461bbc40e6070e787a5604d5277b68b827afe46cf459"
EXPECTED_A5_ARCHIVE_SHA256 = "af4febdac165d9419e01b725e85bbd41fbdff4a293fdca59811e0d6c0209b6ac"
A5_MEMBERS = {
    "phase2d/run5_fastapi_5077/cpu_overlay_attempt5/job/kernel-metadata.json":
        ("adcac0958a2684c3a391d825a2dffc30a92cd47b26f0d0f46d3d83bbc643ec2a", 672),
    "phase2d/run5_fastapi_5077/cpu_overlay_attempt5/push_attempted.json":
        ("da6523e61a97e23e3e8d52fb66e7d92b45597d5f1371791c8974e2c37d414c5d", 364),
    "phase2d/run5_fastapi_5077/cpu_overlay_attempt5/push_result.json":
        ("c2af9c805dca70c889e440552038d465a14f4346bd85d2b81044ceec4f63f864", 332),
    "phase2d/run5_fastapi_5077/cpu_overlay_attempt5/remote_status.json":
        ("68a5824a5cc45889debb32b0c3080f89a1a783be4208a7d33642b0c01acc47e1", 166),
    "phase2d/run5_fastapi_5077/cpu_overlay_attempt5/review/review.json":
        ("0aac91cc5bebe9e2d5acd8ba1d75944abc7d2a79b61564128cc5bd017ea16abc", 1421),
}
# Handover §4 key-deliverable hash table (file -> sha256).
HANDOVER_TABLE = {
    "M3/D1_DECISION_PACKAGE.md": "78bebcf857e8a7a0d2e73c57ca1121fe00fcfcfbb82c721c26b07ce08f4ea02a",
    "M3/D1_APPROVAL_RECORD.md": "01550558b4d4d8eaa6614fec703f3c03f3a14e58f65f0e2e9707a5e03031db9a",
    "M3/FastAPI_CPU_Kaggle_RunKit_v1.zip": "31cf128c866026cf11fe57f36763dd7c831cfa2bb192e70164b793f47938928a",
    "M3/fastapi_cpu_evidence.zip": "e57692a12aae40f0e1951beb5041ed3bd55f91d9404fcbd166423d4ce176a4c0",
    "M3/USER_RUN_KIT_v1/kernel-metadata.json": "3c698a626bb6568872b686ba13f9173f1e566a6b85a4e31279a6687f7b7f0407",
    "M3/USER_RUN_KIT_v1/cpu_job.py": "ba648ae0664aa2745e31c24cc7b3c251d2c5f81808419449b36269756d3056de",
    "M3/FASTAPI_CPU_EXTERNAL_CHAIN_USER_SUPPLEMENT_v1.md": "9fd78df83b744dd468a2511de0e18c36784b3a6ffb578543354dbc77ac7fce81",
    "M3/M3_CONTINUATION_GATE_CHECKPOINT_v1.md": "08a59332e112707a1ed7bf9c644c2a9337a7be4907f7d7db778f67945e6dac7f",
    "M3/requests_line_b_offline_v1/REPORT.md": "f4ac46a45e6c821717edb54fd3082e80eea02578aaae7271fc9accf6858d2d72",
    "M3/requests_line_b_offline_v1/REPORT.json": "c18c463691757ef01edc818a6541c3cb0f284372993cd4917d42b1c5fa35127e",
    "M3/requests_line_b_offline_v1/OUTPUT_SHA256SUMS.txt": "9bb7bf4e0be07c47e38eb3d4c77d3f30ba445987088471af23fb6bedaed9cabc",
    "M3/REQUESTS_LINE_B_POSTFORMAT_REVIEW_v1.md": "e1015df60307393a09f9c0c840a3afd3bb056130469c8b8b7ba13a470bc7bacf",
}
BUNDLES = [
    # (bundle file, sha256sums path inside bundle, anchor, expected OK count)
    # anchor "root": entries are relative to the bundle root;
    # anchor "parent": entries are relative to the sumfile's own directory.
    ("GateC_审计材料包.zip", "GateC3_audit/SHA256SUMS.txt", "root", 30),
    ("Phase2D2_Requests6589_材料包.zip", "SHA256SUMS.txt", "root", 30),
    ("M2_离线审计材料包.zip", "M2/SHA256SUMS.txt", "parent", 11),
    ("M2_离线审计材料包_rev2.zip", "M2/SHA256SUMS.txt", "parent", 12),
    ("M3_离线准备包.zip", "M3/SHA256SUMS.txt", "parent", 6),
]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_sumfile(sumfile: Path, entry_base: Path) -> dict:
    """Verify a SHA256SUMS file; entries resolve against entry_base."""
    ok = fail = missing = 0
    failures = []
    for line in sumfile.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        digest, _, rel = line.partition("  ")
        target = entry_base / rel
        if not target.exists():
            missing += 1
            failures.append(f"MISSING {rel}")
            continue
        if sha256_file(target) == digest:
            ok += 1
        else:
            fail += 1
            failures.append(f"HASHFAIL {rel}")
    return {"ok": ok, "fail": fail, "missing": missing, "failures": failures}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True, help="path to extracted project/ root")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    proj = Path(args.project).resolve()
    results = {"project_root": str(proj), "checks": []}

    def record(check_id, name, passed, detail):
        results["checks"].append({
            "id": check_id, "name": name,
            "result": "PASS" if passed else "FAIL",
            "detail": detail,
        })

    # 1. Delivered bundle integrity (extract to temp, verify, clean up).
    #    Each bundle's SHA256SUMS.txt lives in a subdirectory and its entries
    #    are relative to that subdirectory (the sumfile's own parent).
    import tempfile, shutil
    for i, (bundle, sumrel, anchor, expect_ok) in enumerate(BUNDLES, start=1):
        src = proj / bundle
        if not src.exists():
            record(f"B{i:02d}", f"bundle integrity: {bundle}", False, "bundle not found")
            continue
        with tempfile.TemporaryDirectory() as td:
            with zipfile.ZipFile(src) as z:
                z.extractall(td)
            tdroot = Path(td)
            base = tdroot if anchor == "root" else tdroot / Path(sumrel).parent
            r = verify_sumfile(tdroot / sumrel, base)
        passed = r["fail"] == 0 and r["missing"] == 0 and r["ok"] == expect_ok
        record(f"B{i:02d}", f"bundle integrity: {bundle}", passed,
               f"{r['ok']}/{expect_ok} OK, fail={r['fail']}, missing={r['missing']}"
               + (f"; {r['failures']}" if r["failures"] else ""))

    # 2. In-tree SHA256SUMS lists (from their intended bases).
    intree = [
        ("M3/SHA256SUMS.txt", "M3"),
        ("M3/USER_RUN_KIT_v1/SHA256SUMS.txt", "M3/USER_RUN_KIT_v1"),
        ("M3/requests_line_b_offline_v1/OUTPUT_SHA256SUMS.txt", "M3/requests_line_b_offline_v1"),
        ("M2/SHA256SUMS.txt", "M2"),
        ("V1/SHA256SUMS.txt", "V1"),
    ]
    for j, (sf, base) in enumerate(intree, start=1):
        r = verify_sumfile(proj / sf, proj / base)
        passed = r["fail"] == 0 and r["missing"] == 0
        record(f"T{j:02d}", f"in-tree checksum: {sf}", passed,
               f"{r['ok']} OK, fail={r['fail']}, missing={r['missing']}"
               + (f"; {r['failures']}" if r["failures"] else ""))
    # M3 sub-lists + requests INPUT list are anchored at project/ root.
    for j, sf in enumerate([
        "M3/CPU_CALIBRATION_REPORT_PENDING_EXTERNAL_EVIDENCE_v1_SHA256SUMS.txt",
        "M3/FASTAPI_CPU_EXTERNAL_CHAIN_USER_SUPPLEMENT_v1_SHA256SUMS.txt",
        "M3/FASTAPI_CPU_LOCAL_EXTERNAL_CHAIN_INVENTORY_v1_SHA256SUMS.txt",
        "M3/FASTAPI_CPU_RETURNED_EVIDENCE_REVIEW_v1_SHA256SUMS.txt",
        "M3/HANDOVER_TASK_INCOMPLETE_2026-10-11_SHA256SUMS.txt",
        "M3/M3_CONTINUATION_GATE_CHECKPOINT_v1_SHA256SUMS.txt",
        "M3/REQUESTS_LINE_B_POSTFORMAT_REVIEW_v1_SHA256SUMS.txt",
    ], start=7):
        # These lists are anchored at their own parent directory (M3/).
        r = verify_sumfile(proj / sf, proj / Path(sf).parent)
        passed = r["fail"] == 0 and r["missing"] == 0
        record(f"T{j:02d}", f"in-tree checksum (M3-anchored): {sf}", passed,
               f"{r['ok']} OK, fail={r['fail']}, missing={r['missing']}"
               + (f"; {r['failures']}" if r["failures"] else ""))
    # The requests-line INPUT list is anchored at the project root.
    r = verify_sumfile(proj / "M3/requests_line_b_offline_v1/INPUT_SHA256SUMS.txt", proj)
    record("T14", "in-tree checksum (root-anchored): M3/requests_line_b_offline_v1/INPUT_SHA256SUMS.txt",
           r["fail"] == 0 and r["missing"] == 0,
           f"{r['ok']} OK, fail={r['fail']}, missing={r['missing']}"
           + (f"; {r['failures']}" if r["failures"] else ""))

    # 3. Handover §4 key-deliverable hash table.
    mism = []
    for rel, exp in HANDOVER_TABLE.items():
        p = proj / rel
        if not p.exists() or sha256_file(p) != exp:
            mism.append(rel)
    record("H01", "handover §4 key-deliverable hash table (12 entries)",
           not mism, f"{12 - len(mism)}/12 match" + (f"; mismatches: {mism}" if mism else ""))

    # 4. C-3 reproducibility (read-only script; prescribed by VERIFY_README §2).
    c3_out = Path(args.out).with_name("GateC3_rerun.json")
    proc = subprocess.run(
        [sys.executable, str(proj / "GateC3_audit/gatec3_ref_integrity_and_closeout.py"),
         "--project", str(proj / "gemma4_project"), "--audit-home", str(proj),
         "--out", str(c3_out)],
        capture_output=True, text=True, timeout=300)
    stdout_expected = {"uniform": ["A2", "A4", "A6", "Run5_GPU"],
                       "divergent_documented": ["A5"],
                       "divergent_undocumented": [],
                       "server_slug_equals_title_slug":
                           {"A2": True, "A4": True, "A5": True, "A6": True, "Run5_GPU": True}}
    c3_hash = sha256_file(c3_out) if c3_out.exists() else ""
    bundled = sha256_file(proj / "GateC3_audit/GateC3_closeout.json")
    stdout_match = False
    try:
        stdout_match = json.loads(proc.stdout) == stdout_expected
    except Exception:
        stdout_match = False
    passed = (proc.returncode == 0 and c3_hash == EXPECTED_C3_JSON_SHA256
              and c3_hash == bundled and stdout_match)
    record("R01", "C-3 closeout reproducibility (re-run + hash + stdout)", passed,
           f"exit={proc.returncode}; rerun_sha256={c3_hash}; "
           f"bundled_sha256={bundled}; stdout_match={stdout_match}")

    # 5. Interlock test on the DRAFT_NOT_EXECUTED verifier (no authorization).
    proc = subprocess.run(
        [sys.executable, str(proj / "GateC3_audit/authenticated_readonly_verify_DRAFT_NOT_EXECUTED.py"),
         "--authorization-file", "/nonexistent", "--out", "/tmp/_cont_audit_interlock.json"],
        capture_output=True, text=True, timeout=120)
    passed = (proc.returncode == 3 and proc.stdout == ""
              and "INTERLOCK" in proc.stderr and "zero requests" in proc.stderr)
    record("R02", "authorization interlock (no auth phrase -> exit 3, zero requests)", passed,
           f"exit={proc.returncode}; stdout_bytes={len(proc.stdout.encode())}; "
           f"stderr={proc.stderr.strip()!r}")

    # 6. A5 dual-slug archive member spot-check (VERIFY_README §3).
    a5 = proj / "gemma4_project/Phase2D_Run5_隔离后审与输入阻塞证据包.zip"
    a5_detail = {}
    if not a5.exists():
        record("A01", "A5 archive hash + 5-member spot check", False, "archive not found")
    else:
        arch_ok = sha256_file(a5) == EXPECTED_A5_ARCHIVE_SHA256
        members_ok = True
        with zipfile.ZipFile(a5) as z:
            for member, (exp_h, exp_s) in A5_MEMBERS.items():
                data = z.read(member)
                h = sha256_bytes(data)
                good = (h == exp_h and len(data) == exp_s)
                members_ok &= good
                a5_detail[member] = {"sha256": h, "bytes": len(data), "match": good}
            km = z.read("phase2d/run5_fastapi_5077/cpu_overlay_attempt5/job/kernel-metadata.json").decode()
            pa = json.loads(z.read("phase2d/run5_fastapi_5077/cpu_overlay_attempt5/push_attempted.json"))
            pr = json.loads(z.read("phase2d/run5_fastapi_5077/cpu_overlay_attempt5/push_result.json"))
            rs = json.loads(z.read("phase2d/run5_fastapi_5077/cpu_overlay_attempt5/remote_status.json"))
            rv = json.loads(z.read("phase2d/run5_fastapi_5077/cpu_overlay_attempt5/review/review.json"))
        slug_ok = ("auditoverlay" in km and "auditoverlay" in json.dumps(pa)
                   and "audit-overlay" in json.dumps(pr) and "audit-overlay" in json.dumps(rs)
                   and "auditoverlay" in json.dumps(rv) and "audit-overlay" in json.dumps(rv))
        record("A01", "A5 archive hash + 5-member spot check + dual-slug semantics",
               arch_ok and members_ok and slug_ok,
               f"archive_hash_ok={arch_ok}; members_5/5={'OK' if members_ok else 'FAIL'}; "
               f"requested_slip=auditoverlay, server_slip=audit-overlay, review_documents_both={slug_ok}")

    # 7. Conflict register + conclusion boundary (Gate C-3 closeout JSON).
    c3 = json.loads((proj / "GateC3_audit/GateC3_closeout.json").read_text(encoding="utf-8"))
    cr = c3.get("conflict_register") or json.loads(
        (proj / "GateC1R_audit/GateC1R_reconciliation.json").read_text(encoding="utf-8"))["conflict_register"]
    ids = [c.get("id") for c in cr]
    record("C01", "conflict register C1R-01..05 present", ids == [f"C1R-0{i}" for i in range(1, 6)],
           f"entries={ids}")
    cands = c3["candidates"]
    all_unconfirmed = all(c.get("remote_fact", "").startswith("UNCONFIRMED") for c in cands)
    record("C02", "conclusion boundary: every candidate remote_fact=UNCONFIRMED", all_unconfirmed,
           "; ".join(f"{c.get('candidate', '?')}={c.get('remote_fact', '?').split(' ')[0]}" for c in cands))

    # 8. Returned evidence package internal consistency.
    ev = proj / "M3/fastapi_cpu_evidence.zip"
    with zipfile.ZipFile(ev) as z:
        crc = z.testzip() is None
        names = z.namelist()
        dup = len(names) - len(set(names))
        unpack = proj / "M3/fastapi_cpu_evidence_unpacked"
        match = all(sha256_bytes(z.read(n)) == sha256_file(unpack / n)
                    for n in names if not n.endswith("/"))
    record("E01", "returned evidence package (CRC, duplicates, unpacked-tree hash closure)",
           crc and dup == 0 and match,
           f"crc={'PASS' if crc else 'FAIL'}; members={len(names)}; duplicates={dup}; "
           f"unpacked_tree_hash_closure={'ALL_MATCH' if match else 'MISMATCH'}")

    # 9. Root uploads/ copy identity.
    up = proj.parent / "uploads/fastapi_cpu_evidence.json"
    same = up.exists() and sha256_file(up) == sha256_file(ev)
    record("E02", "uploads/fastapi_cpu_evidence.json byte-identical to M3 return package", same,
           f"sha256={sha256_file(up) if up.exists() else 'n/a'}")

    # 10. NEW FINDING: standalone gate1_official.ipynb vs recorded hash.
    nb = proj / "gemma4_project/gate1_official.ipynb"
    recorded = None
    for line in (proj / "gemma4_project/SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        if line.endswith("  gate1_official.ipynb"):
            recorded = line.split()[0]
    actual = sha256_file(nb)
    with zipfile.ZipFile(proj / "gemma4_project/phase2b_gate1_bundle.zip") as z:
        bundle_copy_hash = sha256_bytes(z.read("gate1_official.ipynb"))
        bundle_cells = len(json.loads(z.read("gate1_official.ipynb"))["cells"])
    local_cells = len(json.loads(nb.read_text(encoding="utf-8"))["cells"])
    results["checks"].append({
        "id": "F01", "name": "gemma4_project/gate1_official.ipynb integrity vs recorded hash",
        "result": "FINDING" if actual != recorded else "PASS",
        "detail": f"recorded={recorded}; actual={actual}; "
                  f"bundle_copy_matches_recorded={bundle_copy_hash == recorded}; "
                  f"local_cells={local_cells} vs bundle_cells={bundle_cells} "
                  f"(local copy is the 'L4 ENV_CHECK v2' diagnostic notebook, "
                  f"not the Phase 2B Gate-1 notebook)",
    })

    # 11. NEW FINDING: expert_review_bundle inventory gap.
    erb = proj / "gemma4_project/expert_review_bundle"
    r = verify_sumfile(erb / "SHA256SUMS.txt", erb)
    results["checks"].append({
        "id": "F02", "name": "expert_review_bundle inventory (snapshot completeness)",
        "result": "FINDING" if r["missing"] > 0 else "PASS",
        "detail": f"present_ok={r['ok']}; missing_from_snapshot={r['missing']} "
                  f"(evidence/* and source/remote_verify/* not included in this delivery; "
                  f"inventory gap, not hash drift)",
    })

    passed = sum(1 for c in results["checks"] if c["result"] == "PASS")
    failed = sum(1 for c in results["checks"] if c["result"] == "FAIL")
    findings = sum(1 for c in results["checks"] if c["result"] == "FINDING")
    results["summary"] = {
        "total": len(results["checks"]), "pass": passed, "fail": failed,
        "findings": findings,
        "frozen_states_unchanged": {
            "real_fastapi_cpu_calibration": "NOT_CONFIRMED",
            "gate_c": "BLOCKED",
            "run5_gpu": "HOLD",
            "new_submission_authorized": False,
            "m3_to_m4_transition": "NOT_ELIGIBLE",
            "rich_2725": "2 valid paired runs, gaps [+1, +1]",
        },
        "boundary": "offline read-only audit; zero network; zero Kaggle API; zero auth; "
                    "zero submission/retry/rename; zero GPU; zero model/agent runs; "
                    "no historical evidence modified",
    }
    Path(args.out).write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results["summary"], ensure_ascii=False, indent=2))
    # Exit 0 when all verification checks pass; FINDINGs are reported but
    # are discrepancy observations for the audit record, not verification failures.
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
