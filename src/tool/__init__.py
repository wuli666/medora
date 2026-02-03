"""Tools for medical agent."""

from .medical_record import MedicalRecordParserTool, create_medical_record_parser_tool

__all__ = [
    "MedicalRecordParserTool",
    "create_medical_record_parser_tool"
]