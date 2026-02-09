"""Structured output models for medical record processing."""

from pydantic import BaseModel
from typing import List, Optional


class MedicalRecordStructure(BaseModel):
    """Structure for extracted medical record information."""
    diagnoses: List[str]
    medications: List[str]
    risk_flags: List[str]


class MedicalRecordOutput(BaseModel):
    """Standardized output structure for medical record parser."""
    tool: str = "medical_record_parser"
    ok: bool
    data: Optional[dict] = None
    error: Optional[str] = None
    source: str = "medgemma"
    latency_ms: int


class SummaryOutput(BaseModel):
    """Standardized output structure for summary node."""
    diagnoses: List[str]
    medications: List[str]
    risk_flags: List[str]