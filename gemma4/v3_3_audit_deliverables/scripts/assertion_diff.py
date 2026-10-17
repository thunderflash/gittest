"""
V3.3-D: Structured Assertion Diff Diagnostic Parser
Extracts failure locations, expected vs actual values, and stack traces.
Follows the principle: 'Diagnose first, edit second'.
"""

import re
from typing import Dict, Any, Optional, List


class AssertionDiffDiagnoser:
    """Parses pytest assertion failures into structured, actionable repair clues."""

    def diagnose_failure(self, stdout: str, stderr: str) -> Dict[str, Any]:
        """Parse test failure and extract precise mismatch evidence."""
        combined = (stdout or "") + "\n" + (stderr or "")

        # 1. Check for Syntax / Indentation Error
        syntax_err = re.search(r'(SyntaxError|IndentationError): (.*)', combined)
        if syntax_err:
            file_line = re.search(r'File "(.*?)", line ([0-9]+)', combined)
            return {
                "category": "SYNTAX_ERROR",
                "error_type": syntax_err.group(1),
                "message": syntax_err.group(2).strip(),
                "file": file_line.group(1) if file_line else None,
                "line": int(file_line.group(2)) if file_line else None,
                "guidance": "Fix syntax error at exact line without modifying logic."
            }

        # 2. Check for Import / Attribute Error
        import_err = re.search(r'(ImportError|ModuleNotFoundError|AttributeError): (.*)', combined)
        if import_err:
            return {
                "category": "INTERFACE_ERROR",
                "error_type": import_err.group(1),
                "message": import_err.group(2).strip(),
                "guidance": "Check symbol spelling, export name, or required imports."
            }

        # 3. Check for Assertion Failure (Actual vs Expected)
        # Pattern A: assert X == Y
        eq_match = re.search(r'E\s+assert (.*) == (.*)', combined)
        if eq_match:
            actual_str = eq_match.group(1).strip()
            expected_str = eq_match.group(2).strip()

            # Detect type of difference
            diff_type = "GENERAL_EQUALITY"
            if actual_str.isdigit() and expected_str.isdigit():
                diff_type = "NUMERICAL_BOUNDARY"
            elif (actual_str.startswith(("'", '"')) and expected_str.startswith(("'", '"'))):
                diff_type = "STRING_MISMATCH"
            elif "None" in (actual_str, expected_str):
                diff_type = "NONE_CHECK"

            return {
                "category": "ASSERTION_FAILURE",
                "diff_type": diff_type,
                "actual": actual_str,
                "expected": expected_str,
                "guidance": f"Align output directly to expected value '{expected_str}' instead of '{actual_str}'."
            }

        # Pattern B: Detailed diff lines (+ / -)
        diff_lines = re.findall(r'E\s+([+\-?]\s+.*)', combined)
        if diff_lines:
            return {
                "category": "ASSERTION_FAILURE",
                "diff_type": "DETAILED_DIFF",
                "diff_lines": diff_lines[:5],
                "guidance": "Examine +/- diff lines from pytest to identify missing or extraneous data."
            }

        # 4. Fallback: Unparseable failure - fallback to raw stack tail
        tail_lines = [l for l in combined.splitlines() if l.strip().startswith("E ") or "FAILED" in l]
        return {
            "category": "RAW_FAILURE",
            "diff_type": "UNSTRUCTURED",
            "evidence": "\n".join(tail_lines[-8:]) if tail_lines else combined[-300:],
            "guidance": "Review raw pytest failure trace directly."
        }
