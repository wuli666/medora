"""Tests for planner/tooler tool-calling behavior in graph nodes."""

import asyncio
import json
import sys
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, ToolMessage

sys.path.insert(0, str(Path(__file__).parent.parent))
pytest.importorskip("langgraph.types")

from src.graph import nodes


async def _noop_mark_stage(*_args, **_kwargs) -> None:
    return None


def _base_state() -> dict:
    return {
        "run_id": "run-1",
        "raw_text": "患者头痛三天",
        "plan": "先检查，再检索",
        "tools_dispatched": False,
        "planner_tool_attempts": 0,
        "messages": [],
        "search_results": "",
        "merged_analysis": "",
        "tool_skipped": False,
        "has_images": False,
        "reflection": "",
        "iteration": 0,
        "planner_decision": "",
    }


def test_planner_routes_to_tooler_when_tool_calls_present(monkeypatch: pytest.MonkeyPatch) -> None:
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

    monkeypatch.setattr(nodes, "mark_stage", _noop_mark_stage)
    monkeypatch.setattr(nodes, "get_chat_model", lambda *args, **kwargs: FakePlannerLLM())

    result = asyncio.run(nodes.planner_node(_base_state()))
    assert result.goto == "tooler"
    assert result.update["tools_dispatched"] is True
    assert isinstance(result.update["messages"][0], AIMessage)
    assert result.update["messages"][0].tool_calls


def test_planner_retries_when_no_tool_calls_then_fallback_to_no_tool_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeToolBoundLLM:
        async def ainvoke(self, _messages):
            return AIMessage(content="no tools", tool_calls=[])

    class FakePlannerLLM:
        def bind_tools(self, _tools):
            return FakeToolBoundLLM()

    monkeypatch.setattr(nodes, "mark_stage", _noop_mark_stage)
    monkeypatch.setattr(nodes, "get_chat_model", lambda *args, **kwargs: FakePlannerLLM())

    state = _base_state()
    first = asyncio.run(nodes.planner_node(state))
    assert first.goto == "planner"
    assert first.update["planner_tool_attempts"] == 1
    assert first.update.get("tools_dispatched", False) is False
    state.update(first.update)

    second = asyncio.run(nodes.planner_node(state))
    assert second.goto == "planner"
    assert second.update["planner_tool_attempts"] == 2
    assert second.update.get("tools_dispatched", False) is False
    state.update(second.update)

    third = asyncio.run(nodes.planner_node(state))
    assert third.goto == "planner"
    assert third.update["tools_dispatched"] is True
    assert third.update["tool_skipped"] is True
    assert third.update["planner_tool_attempts"] == 3
    assert third.update["search_results"] == "（未执行检索，基于已有信息生成）"
    assert third.update["merged_analysis"] == state["raw_text"]


def test_tooler_node_no_tool_calls_is_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(nodes, "mark_stage", _noop_mark_stage)
    state = {"run_id": "run-1", "has_images": False, "messages": [AIMessage(content="hi")]}
    result = asyncio.run(nodes.tooler_node(state))
    assert result == {}


def test_tooler_node_executes_real_tool_calls_only(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeTool:
        async def ainvoke(self, _args):
            return json.dumps(
                {
                    "tool": "analyze_medical_text",
                    "ok": True,
                    "data": {"analysis_text": "文本分析结果"},
                    "error": None,
                    "meta": {"version": "1.0"},
                },
                ensure_ascii=False,
            )

    async def _fake_safe_llm_call(_llm, _prompt, stage: str) -> str:
        if stage == "tooler.search":
            return "检索摘要"
        return "其他摘要"

    monkeypatch.setattr(nodes, "mark_stage", _noop_mark_stage)
    monkeypatch.setattr(nodes, "safe_llm_call", _fake_safe_llm_call)
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
    assert result["merged_analysis"] == "文本分析结果"
    assert result["search_results"] == "检索摘要"
    assert len(result["messages"]) == 1
    assert isinstance(result["messages"][0], ToolMessage)


def _run_reflector_with_payload(monkeypatch: pytest.MonkeyPatch, payload):
    if isinstance(payload, dict):
        payload = nodes.reflect_report(**payload)

    class _StructuredLLM:
        async def ainvoke(self, _messages):
            return payload

    class _ReflectorLLM:
        def with_structured_output(self, _schema):
            return _StructuredLLM()

    monkeypatch.setattr(nodes, "mark_stage", _noop_mark_stage)
    monkeypatch.setattr(nodes, "get_chat_model", lambda *args, **kwargs: _ReflectorLLM())
    state = _base_state()
    state.update(
        {
            "merged_analysis": "分析",
            "search_results": "检索",
            "plan": "计划",
        }
    )
    return asyncio.run(nodes.reflector_node(state))


def test_reflector_structured_output_renders_to_string(monkeypatch: pytest.MonkeyPatch) -> None:
    result = _run_reflector_with_payload(
        monkeypatch,
        {
            "completeness_check": "无明显缺失",
            "consistency_check": "未见明显冲突",
            "hallucination_risk": "未见明显幻觉风险",
            "minimal_corrections": ["继续监测血压变化"],
            "quality_conclusion": "PASS",
        },
    )
    reflection = result["reflection"]
    assert isinstance(reflection, str)
    assert "1. 完整性检查：" in reflection
    assert "2. 一致性检查：" in reflection
    assert "3. 幻觉风险：" in reflection
    assert "4. 最小修正建议：" in reflection
    assert "5. 质检结论：PASS" in reflection


def test_reflector_writes_pass_fail_to_reflection(monkeypatch: pytest.MonkeyPatch) -> None:
    pass_result = _run_reflector_with_payload(
        monkeypatch,
        {
            "completeness_check": "无明显缺失",
            "consistency_check": "未见明显冲突",
            "hallucination_risk": "未见明显幻觉风险",
            "minimal_corrections": ["保持当前管理"],
            "quality_conclusion": "PASS",
        },
    )
    assert "5. 质检结论：PASS" in pass_result["reflection"]

    fail_result = _run_reflector_with_payload(
        monkeypatch,
        {
            "completeness_check": "缺失近期血压记录",
            "consistency_check": "计划与检索建议不一致",
            "hallucination_risk": "部分结论证据不足",
            "minimal_corrections": ["补齐血压日志"],
            "quality_conclusion": "FAIL",
        },
    )
    assert "5. 质检结论：FAIL" in fail_result["reflection"]


def test_reflector_redo_when_fail_and_under_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    result = _run_reflector_with_payload(
        monkeypatch,
        {
            "completeness_check": "缺失近期血压记录",
            "consistency_check": "计划与检索建议不一致",
            "hallucination_risk": "部分结论证据不足",
            "minimal_corrections": ["补齐血压日志"],
            "quality_conclusion": "FAIL",
        },
    )
    assert result["planner_decision"] == "REDO"
    assert result["iteration"] == 1


def test_reflector_summary_when_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    result = _run_reflector_with_payload(
        monkeypatch,
        {
            "completeness_check": "无明显缺失",
            "consistency_check": "未见明显冲突",
            "hallucination_risk": "未见明显幻觉风险",
            "minimal_corrections": ["维持当前随访"],
            "quality_conclusion": "PASS",
        },
    )
    assert result["planner_decision"] == "SUMMARY"


def test_reflector_summary_when_limit_reached(monkeypatch: pytest.MonkeyPatch) -> None:
    class _StructuredLLM:
        async def ainvoke(self, _messages):
            return nodes.reflect_report(
                completeness_check="缺失近期血压记录",
                consistency_check="计划与检索建议不一致",
                hallucination_risk="部分结论证据不足",
                minimal_corrections=["补齐血压日志"],
                quality_conclusion="FAIL",
            )

    class _ReflectorLLM:
        def with_structured_output(self, _schema):
            return _StructuredLLM()

    monkeypatch.setattr(nodes, "mark_stage", _noop_mark_stage)
    monkeypatch.setattr(nodes, "get_chat_model", lambda *args, **kwargs: _ReflectorLLM())

    state = _base_state()
    state.update(
        {
            "iteration": 1,
            "merged_analysis": "分析",
            "search_results": "检索",
            "plan": "计划",
        }
    )
    result = asyncio.run(nodes.reflector_node(state))
    assert result["iteration"] == 2
    assert result["planner_decision"] == "SUMMARY"


def test_planner_routes_redo_to_replan(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(nodes, "mark_stage", _noop_mark_stage)
    monkeypatch.setattr(nodes, "get_chat_model", lambda *args, **kwargs: object())
    state = _base_state()
    state.update(
        {
            "tools_dispatched": True,
            "reflection": "已有反思",
            "planner_decision": "REDO",
        }
    )
    result = asyncio.run(nodes.planner_node(state))
    assert result.goto == "planner"
    assert result.update["planner_decision"] == ""
    assert result.update["tools_dispatched"] is False
    assert result.update["planner_tool_attempts"] == 0
    assert result.update["tool_skipped"] is False


def test_planner_routes_summary_when_reflector_approved(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(nodes, "mark_stage", _noop_mark_stage)
    monkeypatch.setattr(nodes, "get_chat_model", lambda *args, **kwargs: object())
    state = _base_state()
    state.update(
        {
            "tools_dispatched": True,
            "reflection": "已有反思",
            "planner_decision": "SUMMARY",
        }
    )
    result = asyncio.run(nodes.planner_node(state))
    assert result.goto == "summarize"


def test_planner_routes_reflector_on_unknown_decision(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(nodes, "mark_stage", _noop_mark_stage)
    monkeypatch.setattr(nodes, "get_chat_model", lambda *args, **kwargs: object())
    state = _base_state()
    state.update(
        {
            "tools_dispatched": True,
            "planner_decision": "UNKNOWN",
        }
    )
    result = asyncio.run(nodes.planner_node(state))
    assert result.goto == "reflector"
    assert result.update["planner_decision"] == "REFLECT"


def test_planner_toolcall_prompt_includes_reflection(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = {"human": ""}

    class FakeToolBoundLLM:
        async def ainvoke(self, messages):
            captured["human"] = str(messages[-1].content)
            return AIMessage(content="", tool_calls=[])

    class FakePlannerLLM:
        def bind_tools(self, _tools):
            return FakeToolBoundLLM()

    monkeypatch.setattr(nodes, "mark_stage", _noop_mark_stage)
    monkeypatch.setattr(nodes, "get_chat_model", lambda *args, **kwargs: FakePlannerLLM())

    state = _base_state()
    state.update(
        {
            "tools_dispatched": False,
            "planner_tool_attempts": 0,
            "reflection": "5. 质检结论：FAIL",
        }
    )
    asyncio.run(nodes.planner_node(state))
    assert "上一轮质检反馈" in captured["human"]
    assert "5. 质检结论：FAIL" in captured["human"]


def test_planner_update_prompt_includes_reflection(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = {"prompt": ""}

    async def _fake_safe_llm_call(_llm, prompt: str, stage: str) -> str:
        if stage == "planner.update":
            captured["prompt"] = prompt
        return "更新后的计划"

    monkeypatch.setattr(nodes, "mark_stage", _noop_mark_stage)
    monkeypatch.setattr(nodes, "get_chat_model", lambda *args, **kwargs: object())
    monkeypatch.setattr(nodes, "safe_llm_call", _fake_safe_llm_call)

    state = _base_state()
    state.update(
        {
            "tools_dispatched": True,
            "planner_decision": "REFLECT",
            "reflection": "4. 最小修正建议：补充检验指标",
        }
    )
    result = asyncio.run(nodes.planner_node(state))
    assert result.goto == "reflector"
    assert "上一轮质检反馈" in captured["prompt"]
    assert "4. 最小修正建议：补充检验指标" in captured["prompt"]
