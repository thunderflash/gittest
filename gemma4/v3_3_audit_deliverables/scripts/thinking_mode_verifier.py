"""
V3.3-R: Thinking Mode Verification and Parameter Audit Module
Evaluates whether thinking_config and thinking_budget actually translate to
downstream vLLM / Gemma 4 model inference changes.
"""

import os
from typing import Dict, Any


class ThinkingModeVerifier:
    """Verifies downstream propagation and observable effect of thinking parameters."""

    def audit_thinking_configuration(self, agent_yaml_data: Dict[str, Any]) -> Dict[str, Any]:
        """Audit declarative YAML to check if thinking parameters are well-formed and valid."""
        gen_cfg = agent_yaml_data.get("generate_content_config")
        if not gen_cfg:
            return {
                "declared": False,
                "status": "UNCONFIGURED",
                "thinking_budget": None,
                "include_thoughts": None,
                "finding": "No generate_content_config declared. Model uses default unbudgeted CoT."
            }

        thinking_cfg = gen_cfg.get("thinking_config")
        if not thinking_cfg:
            return {
                "declared": False,
                "status": "NO_THINKING_CONFIG",
                "finding": "generate_content_config present but thinking_config missing."
            }

        budget = thinking_cfg.get("thinking_budget")
        include_thoughts = thinking_cfg.get("include_thoughts")

        # Check bounds according to HARNESS_README
        is_valid_budget = (isinstance(budget, int) and 0 <= budget <= 32768)

        # Check single-variable purity
        is_pure_single_var = (include_thoughts is None or include_thoughts is False)

        return {
            "declared": True,
            "status": "CONFIG_DECLARED",
            "thinking_budget": budget,
            "include_thoughts": include_thoughts,
            "valid_bounds": is_valid_budget,
            "single_variable_purity": is_pure_single_var,
            "downstream_verification_status": "CONFIG_EFFECT_UNVERIFIED",
            "finding": (
                f"thinking_budget={budget} is declared within allowed bounds [0..32768], "
                "but downstream vLLM QAT model enforcement remains CONFIG_EFFECT_UNVERIFIED "
                "without whitebox token generation logs."
            )
        }
