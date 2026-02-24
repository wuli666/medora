"""Tests for graph nodes structured output + message pipeline behavior."""

import asyncio
import json
import sys
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, ToolMessage

sys.path.insert(0, str(Path(__file__).parent.parent))
pytest.importorskip("langgraph.types")

from src.graph import nodes
from src.graph.state import CarePlan, MergedMedicalAnalysis, PatientSummary, ReflectReport, SearchSummary


async def _noop_mark_stage(*_args, **_kwargs) -> None:
    return None


def _base_state() -> dict:
    return {
        "run_id": "run-1",
        "raw_text": "患者头痛三天",
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


def test_planner_routes_to_tooler_with_structured_plan(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeToolBoundLLM:
        async def ainvoke(self, _messages):
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "web_search",
                        "args": {"query": "头痛 鉴别诊断"},
                        "id": "call_1",
                        "type": "tool_call",
                    }
                ],
            )

    class FakePlannerLLM:
        def bind_tools(self, _tools):
            return FakeToolBoundLLM()

    async def _fake_invoke_structured(_llm, schema, _messages):
        if schema is CarePlan:
            return (
                CarePlan(
                    condition_analysis=["存在持续头痛"],
                    monitoring_metrics=["记录头痛频率"],
                    lifestyle_advice=["规律作息"],
                ),
                AIMessage(content="care_plan_struct"),
            )
        raise AssertionError("unexpected schema")

    monkeypatch.setattr(nodes, "mark_stage", _noop_mark_stage)
    monkeypatch.setattr(nodes, "get_chat_model", lambda *args, **kwargs: FakePlannerLLM())
    monkeypatch.setattr(nodes, "_invoke_structured", _fake_invoke_structured)

    result = asyncio.run(nodes.planner_node(_base_state()))
    assert result.goto == "tooler"
    assert result.update["tools_dispatched"] is True
    assert isinstance(result.update["plan_struct"], dict)
    assert "condition_analysis" in result.update["plan_struct"]
    assert len(result.update["messages"]) == 2
    assert isinstance(result.update["messages"][0], AIMessage)
    assert isinstance(result.update["messages"][1], AIMessage)
    assert result.update["messages"][1].tool_calls


def test_planner_fallback_without_tool_calls_keeps_struct(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeToolBoundLLM:
        async def ainvoke(self, _messages):
            return AIMessage(content="no tools", tool_calls=[])

    class FakePlannerLLM:
        def bind_tools(self, _tools):
            return FakeToolBoundLLM()

    async def _fake_invoke_structured(_llm, schema, _messages):
        assert schema is CarePlan
        return (
            CarePlan(
                condition_analysis=["轻度头痛"],
                monitoring_metrics=["监测持续时间"],
                lifestyle_advice=["补水休息"],
            ),
            AIMessage(content="care_plan_struct"),
        )

    monkeypatch.setattr(nodes, "mark_stage", _noop_mark_stage)
    monkeypatch.setattr(nodes, "get_chat_model", lambda *args, **kwargs: FakePlannerLLM())
    monkeypatch.setattr(nodes, "_invoke_structured", _fake_invoke_structured)

    result = asyncio.run(nodes.planner_node(_base_state()))
    assert result.goto == "reflector"
    assert result.update["tools_dispatched"] is True
    assert result.update["tool_skipped"] is True
    assert result.update["planner_tool_attempts"] == 3
    assert isinstance(result.update["plan_struct"], dict)
    assert len(result.update["messages"]) >= 2


def test_tooler_node_maps_structured_fields_and_messages(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeTool:
        async def ainvoke(self, _args):
            return json.dumps(
                {
                    "tool": "analyze_medical_text",
                    "ok": True,
                    "data": {
                        "analysis_text": "文本分析结果",
                        "analysis_struct": {
                            "patient_profile": ["成人"],
                            "main_diagnoses": ["紧张性头痛"],
                            "medications": [],
                            "abnormal_indicators": [],
                            "risk_assessment": ["低风险"],
                            "follow_up_points": ["观察48小时"],
                        },
                    },
                    "error": None,
                    "meta": {"version": "1.0"},
                },
                ensure_ascii=False,
            )

    async def _fake_invoke_structured(_llm, schema, _messages):
        if schema is SearchSummary:
            return (
                SearchSummary(
                    key_points=["头痛需排查危险信号"],
                    source_notes=["指南A"],
                    evidence_level="中",
                ),
                AIMessage(content="search_struct"),
            )
        if schema is MergedMedicalAnalysis:
            return (
                MergedMedicalAnalysis(
                    primary_findings=["头痛"],
                    key_abnormalities=[],
                    risk_assessment=["低"],
                    attention_points=["持续加重需就医"],
                ),
                AIMessage(content="merge_struct"),
            )
        raise AssertionError("unexpected schema")

    monkeypatch.setattr(nodes, "mark_stage", _noop_mark_stage)
    monkeypatch.setattr(nodes, "_invoke_structured", _fake_invoke_structured)
    monkeypatch.setattr(nodes, "get_chat_model", lambda *args, **kwargs: object())
    monkeypatch.setattr(nodes, "tools_by_name", {"analyze_medical_text": FakeTool()})

    ai_message = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "analyze_medical_text",
                "args": {"medical_text": "头痛"},
                "id": "call_1",
                "type": "tool_call",
            }
        ],
    )
    state = {
        "run_id": "run-1",
        "has_images": False,
        "messages": [ai_message],
        "raw_text": "头痛",
    }
    result = asyncio.run(nodes.tooler_node(state))
    assert result["medical_text_analysis"] == "文本分析结果"
    assert isinstance(result["medical_text_analysis_struct"], dict)
    assert isinstance(result["merged_analysis_struct"], dict)
    assert isinstance(result["search_results_struct"], dict)
    assert len(result["messages"]) == 2
    assert isinstance(result["messages"][0], ToolMessage)
    assert isinstance(result["messages"][1], AIMessage)


def test_reflector_writes_struct_and_ai_message(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _fake_invoke_structured(_llm, schema, _messages):
        assert schema is ReflectReport
        return (
            ReflectReport(
                completeness_check="无明显缺失",
                consistency_check="未见明显冲突",
                hallucination_risk="未见明显幻觉风险",
                minimal_corrections=["继续监测"],
                quality_conclusion="PASS",
            ),
            AIMessage(content="reflect_struct"),
        )

    monkeypatch.setattr(nodes, "mark_stage", _noop_mark_stage)
    monkeypatch.setattr(nodes, "_invoke_structured", _fake_invoke_structured)
    monkeypatch.setattr(nodes, "get_chat_model", lambda *args, **kwargs: object())

    state = _base_state()
    state.update({"merged_analysis": "分析", "search_results": "检索", "plan": "计划"})
    result = asyncio.run(nodes.reflector_node(state))
    assert isinstance(result["reflection_struct"], dict)
    assert "5. 质检结论：PASS" in result["reflection"]
    assert result["planner_decision"] == "SUMMARY"
    assert isinstance(result["messages"][0], AIMessage)


def test_summarize_node_writes_summary_struct_and_ai_message(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _fake_invoke_structured(_llm, schema, _messages):
        assert schema is PatientSummary
        return (
            PatientSummary(
                report_title="健康管理与随访报告",
                brief_summary="近期以头痛为主，当前建议先规律休息并持续观察变化。",
                key_findings=["近期头痛"],
                medication_reminders=["布洛芬 0.2g，饭后按需服用，日内不超过说明书剂量。"],
                follow_up_tips=["若头痛持续3天以上或明显加重，请在3天内复查。"],
            ),
            AIMessage(content="summary_struct"),
        )

    monkeypatch.setattr(nodes, "mark_stage", _noop_mark_stage)
    monkeypatch.setattr(nodes, "_invoke_structured", _fake_invoke_structured)
    monkeypatch.setattr(nodes, "get_chat_model", lambda *args, **kwargs: object())

    state = _base_state()
    state.update(
        {
            "query_intent": "MEDICAL",
            "merged_analysis": "分析",
            "search_results": "检索",
            "plan": "计划",
            "reflection": "通过",
        }
    )
    result = asyncio.run(nodes.summarize_node(state))
    assert isinstance(result["summary_struct"], dict)
    assert "# 健康管理与随访报告" in result["summary"]
    assert "## 摘要" in result["summary"]
    assert "## 关键发现" in result["summary"]
    assert "## 用药提醒" in result["summary"]
    assert "## 随访提示" in result["summary"]
    assert isinstance(result["messages"][0], AIMessage)


def test_summarize_non_medical_uses_system_human_path(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeLLM:
        async def ainvoke(self, messages):
            assert messages[0].type == "system"
            assert messages[1].type == "human"
            return AIMessage(content="你好，这里是通用回答")

    monkeypatch.setattr(nodes, "mark_stage", _noop_mark_stage)
    monkeypatch.setattr(nodes, "get_chat_model", lambda *args, **kwargs: FakeLLM())

    state = _base_state()
    state.update({"query_intent": "NON_MEDICAL", "raw_text": "你好"})
    result = asyncio.run(nodes.summarize_node(state))
    assert result["summary"] == "你好，这里是通用回答"
    assert isinstance(result["summary_struct"], dict)
    assert isinstance(result["messages"][0], AIMessage)
