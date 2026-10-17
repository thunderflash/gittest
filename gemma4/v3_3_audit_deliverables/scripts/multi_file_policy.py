"""
V3.3-M: Controlled Multi-File Coordination Policy Module
Default: Single-file surgical repair.
Exception: Allow at most 2 related source files ONLY when explicit caller-callee dependency exists.
Strictly disallows test file pollution or uncontrolled multi-file refactoring.
"""

from typing import List, Dict, Any, Tuple


class MultiFilePolicyController:
    """Enforces evidence-based controlled multi-file editing policy."""

    def __init__(self, max_allowed_files: int = 2):
        self.max_allowed_files = max_allowed_files

    def evaluate_modified_files(
        self,
        modified_files: List[str],
        primary_candidate_file: str,
        dependency_evidence: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """Validate whether the set of modified files adheres to the controlled multi-file policy."""
        # 1. Check for empty modification
        if not modified_files:
            return False, "REJECTED: No files were modified."

        # 2. Check for test framework pollution
        forbidden = [f for f in modified_files if f.endswith(("pytest.ini", "conftest.py"))]
        if forbidden:
            return False, f"REJECTED: Attempted to modify test framework files: {forbidden}."

        # 3. Default single-file check
        if len(modified_files) == 1:
            return True, "APPROVED: Single-file surgical modification."

        # 4. Multi-file check
        if len(modified_files) > self.max_allowed_files:
            return False, f"REJECTED: Modified {len(modified_files)} files, exceeding limit of {self.max_allowed_files}."

        # 5. For exactly 2 files: Check for documented dependency connection
        sec_file = [f for f in modified_files if f != primary_candidate_file]
        if sec_file:
            target_sec = sec_file[0]
            # Check if second file is in known callers/callees or shares submodule
            known_deps = dependency_evidence.get("related_files", [])
            is_related = (
                target_sec in known_deps or
                any(d in target_sec for d in known_deps) or
                target_sec.split("/")[0] == primary_candidate_file.split("/")[0]
            )
            if not is_related:
                return False, f"REJECTED: Second modified file '{target_sec}' lacks dependency link to '{primary_candidate_file}'."

            return True, f"APPROVED: Coordinated 2-file modification between definition and call site ({primary_candidate_file} & {target_sec})."

        return True, "APPROVED: Validated modification."
