"""Tests for prompt template loading."""

import importlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.prompts import prompts


PROMPT_CONSTANTS = [
    "MEDGEMMA_TEXT_PROMPT",
    "MEDGEMMA_IMAGE_PROMPT",
    "MERGE_PROMPT",
    "SEARCH_SUMMARY_PROMPT",
    "PLAN_PROMPT",
    "PLAN_INIT_PROMPT",
    "PLAN_TOOL_PROMPT",
    "REFLECT_PROMPT",
    "SUMMARIZE_PROMPT",
    "NON_MEDICAL_REPLY_PROMPT",
    "SUPERVISOR_EVAL_PROMPT",
    "PLANNER_DECIDE_PROMPT",
    "INTENT_CLASSIFY_PROMPT",
]


def test_all_prompt_constants_loaded() -> None:
    for name in PROMPT_CONSTANTS:
        value = getattr(prompts, name)
        assert isinstance(value, str)
        assert value


def test_prompt_placeholders_preserved() -> None:
    assert "{medical_text}" in prompts.MEDGEMMA_TEXT_PROMPT
    assert "{clinical_context}" in prompts.MEDGEMMA_IMAGE_PROMPT
    assert "{analysis}" in prompts.PLAN_PROMPT
    assert "{search_results}" in prompts.PLAN_PROMPT
    assert "{analysis}" in prompts.SUMMARIZE_PROMPT
    assert "{search_results}" in prompts.SUMMARIZE_PROMPT
    assert "{plan}" in prompts.SUMMARIZE_PROMPT
    assert "{reflection}" in prompts.SUMMARIZE_PROMPT


def test_import_fails_when_file_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    missing_dir = Path("/tmp/missing-prompts-dir-for-test")
    monkeypatch.setattr(prompts, "_PROMPTS_DIR", missing_dir)
    with pytest.raises(RuntimeError) as exc:
        prompts._load_prompt_file("plan.md")
    assert "Failed to load prompt file:" in str(exc.value)
    assert str(missing_dir / "plan.md") in str(exc.value)

    importlib.reload(prompts)
