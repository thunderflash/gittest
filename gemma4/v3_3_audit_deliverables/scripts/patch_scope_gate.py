#!/usr/bin/env python3
"""
V3.4 Patch Scope Gate (PSG) Implementation.
Guarantees:
1. Strict Single-File Invariant by default.
2. Protection of test frameworks, build configs, and setup files.
3. Strict dependency verification for any 2nd file modification.
4. Diff line count ceiling protection (< 50 lines).
"""

import os
from typing import List, Dict, Any, Tuple


class PatchScopeGate:
    PROTECTED_PATTERNS = [
        "conftest.py",
        "pytest.ini",
        "tox.ini",
        "setup.py",
        "pyproject.toml",
        "requirements.txt",
        "tests/",
        "test/",
        "testing/"
    ]

    def __init__(self, max_diff_lines: int = 50, max_files: int = 1):
        self.max_diff_lines = max_diff_lines
        self.max_files = max_files

    def evaluate_patch(
        self,
        modified_files: List[str],
        diff_text: str,
        cross_file_dependency_verified: bool = False
    ) -> Tuple[bool, str, Dict[str, Any]]:
        evidence = {
            "modified_files": modified_files,
            "file_count": len(modified_files),
            "diff_lines": len(diff_text.strip().splitlines()),
            "protected_file_violations": [],
            "gate_passed": False
        }

        # 1. Protected path check
        for f in modified_files:
            for prot in self.PROTECTED_PATTERNS:
                if prot in f:
                    evidence["protected_file_violations"].append(f)

        if evidence["protected_file_violations"]:
            return False, f"REJECTED: Modification touches protected files: {evidence['protected_file_violations']}", evidence

        # 2. File count check
        if len(modified_files) > self.max_files:
            if len(modified_files) == 2 and cross_file_dependency_verified:
                # Allowed only under verified cross-file dependency
                pass
            else:
                return False, f"REJECTED: Modified {len(modified_files)} files, exceeding single-file limit ({self.max_files})", evidence

        # 3. Diff size check
        if evidence["diff_lines"] > self.max_diff_lines:
            return False, f"REJECTED: Diff size ({evidence['diff_lines']} lines) exceeds maximum threshold ({self.max_diff_lines})", evidence

        evidence["gate_passed"] = True
        return True, "APPROVED: Patch strictly satisfies scope boundaries", evidence


if __name__ == "__main__":
    psg = PatchScopeGate()
    # Test valid patch
    files = ["fastapi/dependencies/utils.py"]
    diff = "\n".join(["+ line"] * 12)
    ok, msg, ev = psg.evaluate_patch(files, diff)
    print("PSG Test Run (Valid):", ok, msg)

    # Test protected violation
    bad_files = ["tests/test_foo.py"]
    ok2, msg2, ev2 = psg.evaluate_patch(bad_files, diff)
    print("PSG Test Run (Protected):", ok2, msg2)
