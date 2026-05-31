"""Google Gemini LM factory functions for DSPy."""

from __future__ import annotations

import os

import dspy


def configure_lm() -> dspy.LM:
    """Create and register the main orchestration LM (set as DSPy default)."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is not set.")

    model_name = os.environ.get("GEMINI_MODEL", "gemini-2.5-pro")
    lm = dspy.LM(
        f"gemini/{model_name}",
        api_key=api_key,
        cache=False,
    )
    dspy.configure(lm=lm)
    return lm


def configure_sub_lm() -> dspy.LM:
    """Create the sub-LM used for cheap repetitive extractions inside the REPL loop.

    Uses GEMINI_SUB_LM_MODEL if set; falls back to gemini-2.5-flash.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is not set.")

    model_name = os.environ.get("GEMINI_SUB_LM_MODEL", "gemini-2.5-flash")
    return dspy.LM(
        f"gemini/{model_name}",
        api_key=api_key,
        cache=False,
    )
