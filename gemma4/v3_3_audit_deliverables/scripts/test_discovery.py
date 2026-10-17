"""
V3.3-T: Progressive Micro-Targeted Test Discovery Module
Implements 4-tier progressive test node localization:
Tier 1: Explicit test method/function (pytest path/to/test.py -k "test_func" -q)
Tier 2: Target test file (pytest path/to/test.py -q)
Tier 3: Narrow submodule test dir (pytest tests/submodule/ -q)
Tier 4: Guard against empty test collection, syntax errors, and broad suite timeout.
"""

import os
import re
from typing import Dict, Any, List, Optional


class ProgressiveTestDiscovery:
    """Discovers the most targeted test node without running broad test suites."""

    def __init__(self, repo_root: str):
        self.repo_root = repo_root

    def discover_test_target(
        self,
        problem_statement: str,
        target_file: str,
        target_symbol: str,
        existing_test_files: List[str]
    ) -> Dict[str, Any]:
        """Progressively resolve the narrowest test execution command."""
        # 1. Tier 1: Check if problem statement explicitly mentions a test function
        test_func_matches = re.findall(r'(test_[a-zA-Z0-9_]+)', problem_statement)
        test_file_mentions = [f for f in existing_test_files if os.path.basename(f) in problem_statement]

        if test_func_matches and test_file_mentions:
            return {
                "tier": "TIER_1_FUNCTION",
                "command": f"python -m pytest {test_file_mentions[0]} -k \"{test_func_matches[0]}\" -q",
                "test_file": test_file_mentions[0],
                "filter_keyword": test_func_matches[0],
                "rationale": "Explicit test function and test file found in problem description."
            }

        # 2. Tier 2: Match test file corresponding to the target source file
        base_name = os.path.splitext(os.path.basename(target_file))[0]
        matched_tests = [
            f for f in existing_test_files
            if f"test_{base_name}.py" in f or f"{base_name}_test.py" in f
        ]

        if matched_tests:
            # If we also know the symbol name, try to filter by symbol name
            symbol_short = target_symbol.split(".")[-1]
            if symbol_short and len(symbol_short) > 3:
                return {
                    "tier": "TIER_1_FILTERED_FILE",
                    "command": f"python -m pytest {matched_tests[0]} -k \"{symbol_short}\" -q",
                    "test_file": matched_tests[0],
                    "filter_keyword": symbol_short,
                    "rationale": f"Matched test file for {base_name} with symbol filter {symbol_short}."
                }
            return {
                "tier": "TIER_2_FILE",
                "command": f"python -m pytest {matched_tests[0]} -q",
                "test_file": matched_tests[0],
                "filter_keyword": None,
                "rationale": f"Matched test file for {base_name}."
            }

        # 3. Tier 3: Submodule test directory
        sub_dir = os.path.dirname(target_file)
        test_sub_dir = os.path.join("tests", sub_dir.replace("/", "_"))
        if os.path.exists(os.path.join(self.repo_root, test_sub_dir)):
            return {
                "tier": "TIER_3_SUBDIR",
                "command": f"python -m pytest {test_sub_dir} -q",
                "test_file": test_sub_dir,
                "filter_keyword": None,
                "rationale": f"Targeted submodule test directory {test_sub_dir}."
            }

        # 4. Fallback safe narrow directory
        return {
            "tier": "TIER_4_FALLBACK_FILE",
            "command": f"python -m pytest {existing_test_files[0]} -q" if existing_test_files else "python -m pytest -q",
            "test_file": existing_test_files[0] if existing_test_files else "tests/",
            "filter_keyword": None,
            "rationale": "Safe bounded fallback to primary test file."
        }

    def validate_test_output(self, stdout: str, stderr: str, exit_code: int) -> Dict[str, Any]:
        """Validate that tests were actually collected and not falsely passed on empty."""
        combined = stdout + "\n" + stderr

        # Empty test collection guard
        if "collected 0 items" in combined or "no tests ran" in combined:
            return {
                "valid": False,
                "error": "EMPTY_COLLECTION",
                "message": "Pytest collected 0 items. Target filter was too restrictive or path invalid."
            }

        # Collection error guard
        if "collection error" in combined.lower() or "ERROR collecting" in combined:
            return {
                "valid": False,
                "error": "COLLECTION_ERROR",
                "message": "Pytest encountered import or collection error before running tests."
            }

        # Timeout or hung guard
        if exit_code == 124 or "timed out" in combined.lower():
            return {
                "valid": False,
                "error": "TIMEOUT",
                "message": "Test execution exceeded timeout limit."
            }

        # Standard test run
        passed_match = re.search(r'([0-9]+) passed', combined)
        failed_match = re.search(r'([0-9]+) failed', combined)
        passed_count = int(passed_match.group(1)) if passed_match else 0
        failed_count = int(failed_match.group(1)) if failed_match else 0

        return {
            "valid": True,
            "exit_code": exit_code,
            "passed": passed_count,
            "failed": failed_count,
            "all_passed": (exit_code == 0 and passed_count > 0 and failed_count == 0)
        }
