from __future__ import annotations

from typing import Literal, TypedDict

from pydantic import BaseModel, Field
from langgraph.graph import MessagesState


class MedicalTextAnalysis(BaseModel):
    patient_profile: list[str] = Field(
        default_factory=list,
        description="Basic patient profile facts explicitly mentioned in text for future longitudinal comparison.",
    )
    main_diagnoses: list[str] = Field(
        default_factory=list,
        description="Primary diagnoses or suspected conditions extracted from visit records without replacing clinician diagnosis.",
    )
    medications: list[str] = Field(
        default_factory=list,
        description="Current medications, dosage, frequency, and usage details to support reminder generation.",
    )
    abnormal_indicators: list[str] = Field(
        default_factory=list,
        description="Abnormal test findings, biomarkers, or key indicator deviations for baseline and trend tracking.",
    )
    risk_assessment: list[str] = Field(
        default_factory=list,
        description="Clinically relevant risks inferred from provided evidence only, phrased as cautionary understanding support.",
    )
    follow_up_points: list[str] = Field(
        default_factory=list,
        description="Items requiring follow-up, recheck, monitoring, or clarification in recovery-stage management.",
    )


class MedicalImageAnalysis(BaseModel):
    image_type: str = Field(
        default="",
        description="Detected imaging modality or image type (e.g., CT, X-ray, skin photo) for case documentation.",
    )
    key_findings: list[str] = Field(
        default_factory=list,
        description="Most important image findings supported by visual evidence and suitable for patient-facing explanation.",
    )
    abnormal_regions: list[str] = Field(
        default_factory=list,
        description="Anatomical regions with notable abnormalities for later progression comparison.",
    )
    clinical_implications: list[str] = Field(
        default_factory=list,
        description="Clinical significance of findings with cautious interpretation and no diagnostic replacement claims.",
    )
    recommended_checks: list[str] = Field(
        default_factory=list,
        description="Suggested follow-up examinations or validations supporting staged treatment and rehabilitation follow-up.",
    )


class MergedMedicalAnalysis(BaseModel):
    primary_findings: list[str] = Field(
        default_factory=list,
        description="Unified primary findings merged from text and image analyses for structured storage.",
    )
    key_abnormalities: list[str] = Field(
        default_factory=list,
        description="Consolidated abnormalities most clinically relevant for current status and later change tracking.",
    )
    risk_assessment: list[str] = Field(
        default_factory=list,
        description="Overall risk assessment after combining all available evidence, intended for understanding support not diagnosis substitution.",
    )
    attention_points: list[str] = Field(
        default_factory=list,
        description="Priority concerns, action triggers, and points that need further clinical attention or follow-up.",
    )


class SearchSummary(BaseModel):
    key_points: list[str] = Field(
        default_factory=list,
        description="Case-relevant knowledge points distilled from search evidence to improve patient understanding of terms and context.",
    )
    source_notes: list[str] = Field(
        default_factory=list,
        description="Source clues or references supporting each key point for traceability and future review.",
    )
    evidence_level: str = Field(
        default="",
        description="Overall confidence/evidence strength label for retrieved knowledge, explicitly marking uncertainty when needed.",
    )


class CarePlan(BaseModel):
    condition_analysis: list[str] = Field(
        default_factory=list,
        description="Patient condition interpretation, likely causes, and risk reminders framed for self-management support.",
    )
    monitoring_metrics: list[str] = Field(
        default_factory=list,
        description="Trackable metrics with thresholds or warning signals to support long-term trend monitoring.",
    )
    lifestyle_advice: list[str] = Field(
        default_factory=list,
        description="Practical, sustainable daily-life recommendations that improve treatment adherence and recovery execution.",
    )


class ReflectReport(BaseModel):
    completeness_check: str = Field(
        description="Missing key medical history/exam/medication information."
    )
    consistency_check: str = Field(
        description="Whether analysis, search and plan are consistent."
    )
    hallucination_risk: str = Field(
        description="Claims potentially lacking evidence."
    )
    minimal_corrections: list[str] = Field(
        default_factory=list,
        description="1-3 executable minimal corrections.",
    )
    quality_conclusion: Literal["PASS", "FAIL"] = Field(
        description="Final quality gate conclusion."
    )


class PatientSummary(BaseModel):
    report_title: str = Field(
        default="Health Management & Follow-up Report",
        description="Fixed report title for patient-facing final report.",
    )
    brief_summary: str = Field(
        default="",
        description="A short plain-language summary, roughly around 100 words.",
    )
    key_findings: list[str] = Field(
        default_factory=list,
        description="Patient-facing plain-language key findings from the full workflow with high readability.",
    )
    medication_reminders: list[str] = Field(
        default_factory=list,
        description="Medication name, dose, timing and cautions that patients can execute safely.",
    )
    follow_up_tips: list[str] = Field(
        default_factory=list,
        description="Follow-up timing, trigger conditions, and recheck suggestions.",
    )


class MedAgentState(MessagesState):
    run_id: str
    # Input
    raw_text: str  # original visit-related text input
    images: list[str]  # base64 encoded clinical images or report screenshots
    has_images: bool

    # MedGemma analysis (string fields keep backward compatibility; *_struct for durable storage)
    medical_text_analysis: str
    medical_image_analysis: str
    merged_analysis: str
    medical_text_analysis_struct: dict | None
    medical_image_analysis_struct: list[dict]
    merged_analysis_struct: dict | None

    # Search & knowledge enrichment
    search_results: str
    search_results_struct: dict | None

    # Management plan generation
    plan: str
    plan_struct: dict | None
    planner_decision: str
    tools_dispatched: bool
    planner_tool_attempts: int

    # Reflection / quality gate
    reflection: str
    reflection_struct: dict | None
    iteration: int

    # Final patient-facing output
    summary: str
    summary_struct: dict | None
    tool_skipped: bool

    # Supervisor loop control and routing cache
    query_intent: str  # "MEDICAL" | "NON_MEDICAL"
