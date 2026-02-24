"""Focused tests for structured state double-write behavior."""

import asyncio
import sys
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage

sys.path.insert(0, str(Path(__file__).parent.parent))
pytest.importorskip("langgraph.types")

from src.graph import nodes
from src.graph.state import CarePlan, PatientSummary, ReflectReport


async def _noop_mark_stage(*_args, **_kwargs) -> None:
    return None


def _state() -> dict:
    return {
        "run_id": "r1",
        "raw_text": "患者头痛",
        "plan": "",
        "plan_struct": None,
        "tools_dispatched": False,
        "planner_tool_attempts": 0,
        "messages": [],
        "search_results": "",
        "search_results_struct": None,
        "merged_analysis": "",
        "merged_analysis_struct": None,
        "medical_text_analysis": "",
        "medical_text_analysis_struct": None,
        "medical_image_analysis": "",
        "medical_image_analysis_struct": [],
        "tool_skipped": False,
        "has_images": False,
        "reflection": "",
        "reflection_struct": None,
        "iteration": 0,
        "planner_decision": "",
        "query_intent": "MEDICAL",
        "summary": "",
        "summary_struct": None,
        "images": [],
    }


def test_plan_reflect_summary_double_write(monkeypatch):
    class _NoToolLLM:
        def bind_tools(self, _tools):
            return self

        async def ainvoke(self, _messages):
            return AIMessage(content="no tools", tool_calls=[])

    async def _fake_invoke_structured(_llm, schema, _messages):
        if schema is CarePlan:
            return (
                CarePlan(
                    condition_analysis=["头痛"],
                    monitoring_metrics=["观察"],
                    lifestyle_advice=["休息"],
                ),
                AIMessage(content="plan"),
            )
        if schema is ReflectReport:
            return (
                ReflectReport(
                    completeness_check="无明显缺失",
                    consistency_check="未见明显冲突",
                    hallucination_risk="未见明显幻觉风险",
                    minimal_corrections=["继续监测"],
                    quality_conclusion="PASS",
                ),
                AIMessage(content="reflect"),
            )
        if schema is PatientSummary:
            return (
                PatientSummary(
                    report_title="Health Management & Follow-up Report",
                    brief_summary="当前以头痛症状为主，建议先休息并观察变化。",
                    key_findings=["头痛"],
                    medication_reminders=["按医嘱规律服药，避免自行加量。"],
                    follow_up_tips=["若症状加重，请在3天内复查。"],
                ),
                AIMessage(content="summary"),
            )
        raise AssertionError("unexpected schema")

    monkeypatch.setattr(nodes, "mark_stage", _noop_mark_stage)
    monkeypatch.setattr(nodes, "get_chat_model", lambda *args, **kwargs: _NoToolLLM())
    monkeypatch.setattr(nodes, "_invoke_structured", _fake_invoke_structured)

    state = _state()
    planner_result = asyncio.run(nodes.planner_node(state))
    assert isinstance(planner_result.update["plan_struct"], dict)
    assert isinstance(planner_result.update["plan"], str)

    state.update(planner_result.update)
    reflector_result = asyncio.run(nodes.reflector_node(state))
    assert isinstance(reflector_result["reflection_struct"], dict)
    assert isinstance(reflector_result["reflection"], str)

    state.update(reflector_result)
    summarize_result = asyncio.run(nodes.summarize_node(state))
    assert isinstance(summarize_result["summary_struct"], dict)
    assert isinstance(summarize_result["summary"], str)
