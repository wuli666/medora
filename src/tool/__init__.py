"""Tool exports and registry."""

from src.tool.medgemma_tool import analyze_medical_image, analyze_medical_text
from src.tool.search_tools import rag_search, web_search
from .medical_record import MedicalRecordParserTool, create_medical_record_parser_tool

tools = [
    analyze_medical_text,
    analyze_medical_image,
    web_search,
    rag_search,
]
tools_by_name = {tool.name: tool for tool in tools}

__all__ = [
    "analyze_medical_text",
    "analyze_medical_image",
    "web_search",
    "rag_search",
    "tools",
    "tools_by_name",
    "MedicalRecordParserTool",
    "create_medical_record_parser_tool"
]
