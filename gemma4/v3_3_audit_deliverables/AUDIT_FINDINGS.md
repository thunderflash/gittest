# V3.3 audit bundle extraction and independent review

## Scope
Automated archive-integrity, JSON syntax, ledger/trajectory consistency, and static source review. This workflow does not run the competition harness, pytest, or Gemma inference.

## Results
- Archive entries: 77; extracted files: 71.
- ZIP CRC test: passed.
- JSON files parsed: 46; JSON parse failures: [].
- G1 ledger rows: 45; unique run IDs: 45; ledger IDs missing trajectory files: 0.
- G1 script generates random metrics: True.
- G1 script launches a test subprocess: False.
- Python scripts containing hard-coded /home/user/ paths: ["scripts/generate_manifest.py", "scripts/run_v3_3_ablation_suite.py", "scripts/run_g1_g2_experiments.py"].
- No duplicate ZIP paths or path traversal entries were accepted.

## Findings

### High — G1 measurements are not established as physical test evidence
scripts/run_g1_g2_experiments.py generates run-level values with random functions for timings, collected-item count, turns/tool calls, and session time. The script contains no direct subprocess or os.system test invocation. Although it imports component algorithms, that does not prove the ledger's PASS/timing/collection values came from actual pytest executions. Treat these fields as synthetic or unverified until raw command output, exit codes, and measured durations are supplied.

### High — reproducibility concern
The experiment script seeds with random.seed(seed + hash(cand)). Python's built-in hash() is randomized between processes unless PYTHONHASHSEED is fixed; output may therefore change across runs.

### Medium — machine-specific paths
Hard-coded absolute paths were found in: ["scripts/generate_manifest.py", "scripts/run_v3_3_ablation_suite.py", "scripts/run_g1_g2_experiments.py"]. Derive paths from Path(__file__).resolve() or accept workspace/output directories as CLI arguments.

### Medium — manifest coverage
The manifest generator uses non-recursive glob("*") for category files and summarizes trajectories as a count rather than recording a SHA-256 for every trajectory. Use recursive traversal and verify every file digest.

### Medium — weak multi-file dependency evidence
multi_file_policy.py can accept a second file based on substring matching or merely sharing the same top-level directory. Require an explicit normalized dependency edge and retain the evidence supporting it.

## Decision
Archive structure and consistency checks can pass while experimental claims remain unproven. Do not treat this bundle as proof of improved official score or production readiness. Run real, isolated harness tests before promotion.
