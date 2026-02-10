"""Reflection and verification node for medical assistant responses."""

from __future__ import annotations

import json
import re
from typing import Any, Literal, Optional

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage
from pydantic import BaseModel, Field, model_validator

from src.config import settings
from src.llm import get_llm
from src.graph.state import GraphState
from src.prompts import REFLECT_VERIFY_PROMPT_TEMPLATE


OVERCONFIDENT_PATTERN = re.compile(r"\b(definitely|certainly|guaranteed|always|never)\b")
DIRECT_DIAGNOSIS_PATTERN = re.compile(
    r"\b(you have|you likely have|this is|must take|diagnosed with|caused by)\b"
)
UNSAFE_IMPERATIVE_PATTERN = re.compile(
    r"\b(immediately|demand|must|do not listen|ignore)\b"
)
ANTI_CLINICIAN_PATTERN = re.compile(
    r"\b(do not listen to any doctor|don't listen to any doctor|ignore (?:any |your )?(?:doctor|clinician|medical professional))\b"
)
UNCERTAINTY_PATTERN = re.compile(r"\b(may|might|could|possibly|likely)\b")
CLINICIAN_PATTERN = re.compile(
    r"\b(doctor|clinician|healthcare professional|medical professional)\b"
)

SEVERITY_RANK = {"low": 0, "medium": 1, "high": 2}


IssueType = Literal[
    "insufficient_content",
    "overconfident_language",
    "direct_diagnosis_risk",
    "unsafe_imperative",
    "anti_clinician_guidance",
    "missing_caution_language",
    "tool_execution_failed",
    "missing_evidence",
    "reflection_model_error",
    "reflection_inconsistency",
]


class ReflectIssue(BaseModel):
    """Single issue identified during reflection."""

    type: IssueType = Field(description="Machine-readable issue type.")
    severity: Literal["low", "medium", "high"] = Field(
        description="Issue severity level."
    )
    description: str = Field(description="Agent-readable issue description.")
    location: Optional[str] = Field(
        default=None,
        description="Optional location hint where issue appears.",
    )
    evidence_quote: str = Field(
        default="",
        description="Supporting quote or evidence snippet from draft/tool output.",
    )
    policy_trigger: str = Field(
        default="",
        description="Machine-readable policy trigger identifier.",
    )


class ReflectReport(BaseModel):
    """Structured report returned by the reflection node."""

    issues: list[ReflectIssue] = Field(default_factory=list)
    needs_tool: bool = False
    needs_clarification: bool = False
    confidence: Literal["low", "medium", "high"] = "medium"
    risk_level: Literal["safe", "caution", "unsafe"] = "caution"
    policy_triggers: list[str] = Field(default_factory=list)
    is_safe_to_finalize: bool = False

    @model_validator(mode="after")
    def _apply_consistency_guards(self) -> "ReflectReport":
        """Ensure unsafe conclusions always carry machine-readable issues."""
        has_high = any(issue.severity == "high" for issue in self.issues)
        if has_high:
            self.is_safe_to_finalize = False
            self.risk_level = "unsafe"

        if self.is_safe_to_finalize is False and not self.issues:
            self.issues.append(
                ReflectIssue(
                    type="reflection_inconsistency",
                    severity="low",
                    description="Unsafe conclusion without issues; consistency guard injected.",
                    location="verification_report",
                    evidence_quote="",
                    policy_trigger="consistency_guard",
                )
            )
            if self.risk_level == "safe":
                self.risk_level = "caution"

        if self.risk_level == "unsafe" and not has_high:
            self.risk_level = "caution" if self.issues else "safe"

        return self


def _risk_level_from_issues(issues: list[ReflectIssue]) -> Literal["safe", "caution", "unsafe"]:
    """Map issue severities to overall risk label."""
    if any(issue.severity == "high" for issue in issues):
        return "unsafe"
    if issues:
        return "caution"
    return "safe"


def _message_text(content: Any) -> str:
    """Normalize message content into plain text for prompts."""
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=True)


def _latest_user_question(messages: list[AnyMessage]) -> HumanMessage:
    """Get the latest user question from message history."""
    for message in reversed(messages):
        if message.type == "human":
            if isinstance(message, HumanMessage):
                return message
            return HumanMessage(content=_message_text(message.content))
    return HumanMessage(content="")  # Fallback empty message if none found


def _tool_results_from_messages(messages: list[AnyMessage]) -> list[ToolMessage]:
    """Extract ToolMessage results from history."""
    tool_results: list[ToolMessage] = []
    for message in messages:
        if message.type == "tool" and isinstance(message, ToolMessage):
            tool_results.append(message)

    return tool_results


def _tool_result_payload(tool_message: ToolMessage) -> dict[str, Any]:
    """Normalize compact ToolMessage content to a dictionary payload."""
    payload: dict[str, Any] = {
        "tool_call_id": tool_message.tool_call_id,
        "name": tool_message.name,
    }
    content = tool_message.content
    if isinstance(content, str):
        try:
            parsed = json.loads(content)
            if isinstance(parsed, dict):
                payload.update(parsed)
            else:
                payload["summary"] = content
        except json.JSONDecodeError:
            payload["summary"] = content
    else:
        payload["summary"] = _message_text(content)

    if "ok" not in payload:
        payload["ok"] = True
    return payload


def _rule_based_issues(
    draft_response: str,
    tool_results: list[ToolMessage],
) -> ReflectReport:
    """Run deterministic safety checks on the draft response."""
    issues: list[ReflectIssue] = []
    policy_triggers: set[str] = set()
    lower_text = draft_response.lower()

    if not draft_response.strip():
        issues.append(
            ReflectIssue(
                type="insufficient_content",
                severity="high",
                description="Draft response is empty and cannot be finalized.",
                location="draft_response",
                evidence_quote="",
                policy_trigger="empty_draft",
            )
        )
        policy_triggers.add("empty_draft")
        return ReflectReport(
            issues=issues,
            needs_tool=False,
            needs_clarification=True,
            confidence="low",
            risk_level="unsafe",
            policy_triggers=sorted(policy_triggers),
            is_safe_to_finalize=False,
        )

    overconfident_match = OVERCONFIDENT_PATTERN.search(lower_text)
    if overconfident_match:
        issues.append(
            ReflectIssue(
                type="overconfident_language",
                severity="medium",
                description="Response uses certainty language that may be unsafe medically.",
                location="draft_response",
                evidence_quote=overconfident_match.group(0),
                policy_trigger="certainty_claim",
            )
        )
        policy_triggers.add("certainty_claim")

    diagnosis_match = DIRECT_DIAGNOSIS_PATTERN.search(lower_text)
    if diagnosis_match:
        issues.append(
            ReflectIssue(
                type="direct_diagnosis_risk",
                severity="high",
                description="Response appears to provide a direct diagnosis or imperative treatment.",
                location="draft_response",
                evidence_quote=diagnosis_match.group(0),
                policy_trigger="direct_diagnosis",
            )
        )
        policy_triggers.add("direct_diagnosis")

    unsafe_imperative_match = UNSAFE_IMPERATIVE_PATTERN.search(lower_text)
    if unsafe_imperative_match:
        issues.append(
            ReflectIssue(
                type="unsafe_imperative",
                severity="high",
                description="Response contains unsafe coercive or imperative instructions.",
                location="draft_response",
                evidence_quote=unsafe_imperative_match.group(0),
                policy_trigger="unsafe_command",
            )
        )
        policy_triggers.add("unsafe_command")

    anti_clinician_match = ANTI_CLINICIAN_PATTERN.search(lower_text)
    if anti_clinician_match:
        issues.append(
            ReflectIssue(
                type="anti_clinician_guidance",
                severity="high",
                description="Response discourages professional medical guidance.",
                location="draft_response",
                evidence_quote=anti_clinician_match.group(0),
                policy_trigger="anti_clinician",
            )
        )
        policy_triggers.add("anti_clinician")

    has_uncertainty = bool(UNCERTAINTY_PATTERN.search(lower_text))
    has_clinician_advice = bool(CLINICIAN_PATTERN.search(lower_text))
    if not has_uncertainty and not has_clinician_advice:
        issues.append(
            ReflectIssue(
                type="missing_caution_language",
                severity="medium",
                description="Response should include uncertainty and clinician consultation guidance.",
                location="draft_response",
                evidence_quote="",
                policy_trigger="missing_caution",
            )
        )
        policy_triggers.add("missing_caution")

    tool_payloads = [_tool_result_payload(result) for result in tool_results]
    failed_tools = [result for result in tool_payloads if not result.get("ok", False)]
    if failed_tools:
        issues.append(
            ReflectIssue(
                type="tool_execution_failed",
                severity="medium",
                description="One or more tools failed; answer may need additional evidence.",
                location="tool_results",
                evidence_quote=_message_text(failed_tools[0]) if failed_tools else "",
                policy_trigger="tool_failed",
            )
        )
        policy_triggers.add("tool_failed")

    needs_tool = bool(failed_tools)
    needs_clarification = any(issue.type == "insufficient_content" for issue in issues)
    high_risk = any(issue.severity == "high" for issue in issues)

    return ReflectReport(
        issues=issues,
        needs_tool=needs_tool,
        needs_clarification=needs_clarification,
        confidence="low" if high_risk else "medium",
        risk_level=_risk_level_from_issues(issues),
        policy_triggers=sorted(policy_triggers),
        is_safe_to_finalize=not high_risk and not needs_tool and not needs_clarification,
    )


def _merge_reports(rule_report: ReflectReport, llm_report: Optional[ReflectReport]) -> ReflectReport:
    """Merge rule-based and LLM reports into a single normalized report."""
    if llm_report is None:
        return rule_report

    merged_by_key: dict[tuple[str, Optional[str], str], ReflectIssue] = {}
    for issue in [*rule_report.issues, *llm_report.issues]:
        key = (issue.type, issue.location, issue.evidence_quote)
        existing = merged_by_key.get(key)
        if existing is None or SEVERITY_RANK[issue.severity] > SEVERITY_RANK[existing.severity]:
            merged_by_key[key] = issue
    merged_issues = list(merged_by_key.values())

    confidence_rank = {"low": 0, "medium": 1, "high": 2}
    confidence = (
        llm_report.confidence
        if confidence_rank[llm_report.confidence] >= confidence_rank[rule_report.confidence]
        else rule_report.confidence
    )

    needs_tool = rule_report.needs_tool or llm_report.needs_tool
    needs_clarification = (
        rule_report.needs_clarification or llm_report.needs_clarification
    )
    has_high = any(issue.severity == "high" for issue in merged_issues)
    policy_triggers = sorted(
        set(rule_report.policy_triggers).union(set(llm_report.policy_triggers))
    )
    risk_level = _risk_level_from_issues(merged_issues)
    is_safe_to_finalize = not has_high and not needs_tool and not needs_clarification

    if not is_safe_to_finalize and not merged_issues:
        merged_issues.append(
            ReflectIssue(
                type="reflection_inconsistency",
                severity="low",
                description="Unsafe conclusion without explicit issues; inserted by merger.",
                location="merge_reports",
                evidence_quote="",
                policy_trigger="consistency_guard",
            )
        )
        policy_triggers = sorted(set(policy_triggers).union({"consistency_guard"}))
        if risk_level == "safe":
            risk_level = "caution"

    return ReflectReport(
        issues=merged_issues,
        needs_tool=needs_tool,
        needs_clarification=needs_clarification,
        confidence=confidence,
        risk_level=risk_level,
        policy_triggers=policy_triggers,
        is_safe_to_finalize=is_safe_to_finalize,
    )


async def _llm_reflect_report(
    prompt: str,
    model_name: str,
    base_url: str,
) -> ReflectReport:
    """Run LLM-based reflection with structured output."""
    llm = get_llm(model=model_name, base_url=base_url, temperature=0)
    structured_llm = llm.llm.with_structured_output(ReflectReport)
    result = await structured_llm.ainvoke(prompt)
    if isinstance(result, ReflectReport):
        return result
    return ReflectReport.model_validate(result)


async def _reflect_verify_update(state: GraphState) -> dict[str, Any]:
    """Compute reflection update payload from current graph state.

    Args:
        state: Current graph state.

    Returns:
        State update dictionary with verification report and loop count.
    """
    messages = state.get("messages", [])
    draft_response = state.get("draft_response", "")
    tool_results = _tool_results_from_messages(messages)
    tool_results_payload = [_tool_result_payload(result) for result in tool_results]

    reflect_loop_count = int(state.get("reflect_loop_count", 0)) + 1

    rule_report = _rule_based_issues(draft_response, tool_results)

    llm_report: Optional[ReflectReport] = None
    use_llm_reflection = bool(state.get("use_llm_reflection", True))
    has_high_risk_rule_issue = any(issue.severity == "high" for issue in rule_report.issues)
    if use_llm_reflection and not has_high_risk_rule_issue:
        try:
            model_name = settings.MEDGEMMA_REFLECT_MODEL
            base_url = settings.OLLAMA_BASE_URL
            prompt = REFLECT_VERIFY_PROMPT_TEMPLATE.format(
                question=_message_text(_latest_user_question(messages).content),
                draft_response=draft_response,
                tool_results_json=json.dumps(tool_results_payload, ensure_ascii=True),
            )
            llm_report = await _llm_reflect_report(
                prompt, model_name=model_name, base_url=base_url
            )
        except Exception as exc:
            llm_report = ReflectReport(
                issues=[
                    ReflectIssue(
                        type="reflection_model_error",
                        severity="low",
                        description=(
                            "LLM reflection failed; using rule-based checks only: "
                            f"{type(exc).__name__}"
                        ),
                        location="reflect_verify_node",
                        evidence_quote="",
                        policy_trigger="reflection_error",
                    )
                ],
                needs_tool=False,
                needs_clarification=False,
                confidence="low",
                risk_level="caution",
                policy_triggers=["reflection_error"],
                is_safe_to_finalize=False,
            )

    report = _merge_reports(rule_report, llm_report)

    return {
        "verification_report": report.model_dump(),
        "reflect_loop_count": reflect_loop_count,
    }


async def reflect_verify_node(
    state: GraphState,
) -> dict[str, Any]:
    """LangGraph node that verifies a draft response and updates state."""
    update = await _reflect_verify_update(state)
    report_payload = update["verification_report"]
    reflect_loop_count = int(update["reflect_loop_count"])
    max_reflect_loops = int(state.get("max_reflect_loops", settings.MAX_REFLECT_LOOPS))

    report = ReflectReport.model_validate(report_payload)
    if reflect_loop_count >= max_reflect_loops:
        report = ReflectReport(
            issues=report.issues,
            needs_tool=False,
            needs_clarification=False,
            confidence=report.confidence,
            is_safe_to_finalize=True,
        )

    reflect_message = AIMessage(
        content=(
            "Reflection complete. "
            f"safe={report.is_safe_to_finalize}, "
            f"needs_tool={report.needs_tool}, "
            f"needs_clarification={report.needs_clarification}."
        )
    )

    return {
        "verification_report": report.model_dump(),
        "reflect_loop_count": reflect_loop_count,
        "messages": [reflect_message],
    }


async def reflect_verify(payload: GraphState) -> dict[str, Any]:
    """Convenience function to run reflection standalone.

    Args:
        payload: Input payload with draft and context.

    Returns:
        Verification report dictionary.
    """
    return (await _reflect_verify_update(payload))["verification_report"]
