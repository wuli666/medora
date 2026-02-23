"""
Medical Record Parser Tool (Educational Extractor)
Responsibility:
- Take raw medical record text
- Translate medical jargon into patient-friendly language
- Explain medical conditions, test results, and treatments
- Return standardized output schema with latency and status
"""

import time
import logging
from typing import Dict, Any, Optional
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from ..llm.llm import OllamaLLM
from ..prompts.prompts import (
    STRUCTURED_EXTRACTION_SYSTEM_PROMPT,
    STRUCTURED_EXTRACTION_USER_PROMPT_TEMPLATE,
    PATIENT_EDUCATION_SYSTEM_PROMPT,
    PATIENT_EDUCATION_USER_PROMPT_TEMPLATE
)
from ..agents.output_models import MedicalRecordStructure, MedicalRecordOutput

logger = logging.getLogger(__name__)

class MedicalRecordInput(BaseModel):
    """Input schema for medical record parser tool."""
    medical_record: str = Field(
        ...,
        description="Raw medical record text (doctor notes, discharge summary, EMR, etc.)"
    )

class MedicalRecordOutput(BaseModel):
    """Standardized output schema for medical_record_parser."""
    tool: str = "medical_record_parser"
    ok: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    source: str = "medgemma"
    latency_ms: int


class MedicalRecordParserTool(BaseTool):
    """
    Medical record parsing and patient education tool.
    Translates complex medical jargon into simple, patient-friendly terms.
    """

    name: str = "medical_record_parser"
    description: str = (
        "Translate medical record information into patient-friendly language, "
        "explain medical terms, conditions, test results, and treatments in simple terms."
    )
    args_schema: type = MedicalRecordInput

    # 工具配置
    model: str = Field(default="alibayram/medgemma")
    base_url: str = Field(default="http://localhost:11434")
    temperature: float = Field(default=0.2)
    num_ctx: int = Field(default=8192)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._llm = OllamaLLM(
            model=self.model,
            base_url=self.base_url,
            temperature=self.temperature,
            num_ctx=self.num_ctx,
        )



    def _get_prompts(self, medical_record: str):
        system_prompt = PATIENT_EDUCATION_SYSTEM_PROMPT
        user_prompt = PATIENT_EDUCATION_USER_PROMPT_TEMPLATE.format(
            medical_record=medical_record
        )
        return system_prompt, user_prompt.strip()

    def _run(self, medical_record: str) -> Dict[str, Any]:
        start_time = time.time()
        
        if not medical_record or len(medical_record.strip()) < 10:
            latency = int((time.time() - start_time) * 1000)
            return MedicalRecordOutput(
                ok=False,
                error="Medical record text is too short or empty.",
                latency_ms=latency
            ).model_dump()

        try:
            system_prompt, user_prompt = self._get_prompts(medical_record)
            
            response = self._llm.invoke(
                prompt=user_prompt,
                system_prompt=system_prompt,
            )

            struct_system_prompt = STRUCTURED_EXTRACTION_SYSTEM_PROMPT
            struct_user_prompt = STRUCTURED_EXTRACTION_USER_PROMPT_TEMPLATE.format(
                medical_record=medical_record
            )
            
            structured_llm = self._llm.llm.with_structured_output(MedicalRecordStructure)
            structured_medical_record = structured_llm.invoke([
                {"role": "system", "content": struct_system_prompt},
                {"role": "user", "content": struct_user_prompt}
            ])

            latency = int((time.time() - start_time) * 1000)
            data = {
                "type": "patient_education_parse",
                "raw_text_length": len(medical_record),
                "parsed_content": response,
                "patient_friendly": True,
                "structured_medical_record": {
                    "diagnoses": structured_medical_record.diagnoses,
                    "medications": structured_medical_record.medications,
                    "risk_flags": structured_medical_record.risk_flags
                }
            }
            
            return MedicalRecordOutput(
                ok=True,
                data=data,
                latency_ms=latency
            ).model_dump()

        except Exception as e:
            latency = int((time.time() - start_time) * 1000)
            logger.exception("medical_record_parser unexpected error")
            return MedicalRecordOutput(
                ok=False,
                error=f"LLM invocation failed: {str(e)}",
                latency_ms=latency
            ).model_dump()

    async def _arun(self, medical_record: str) -> Dict[str, Any]:
        start_time = time.time()
        
        try:
            system_prompt, user_prompt = self._get_prompts(medical_record)
            
            response = await self._llm.ainvoke(
                prompt=user_prompt,
                system_prompt=system_prompt,
            )

            struct_system_prompt = STRUCTURED_EXTRACTION_SYSTEM_PROMPT
            struct_user_prompt = STRUCTURED_EXTRACTION_USER_PROMPT_TEMPLATE.format(
                medical_record=medical_record
            )
            
            structured_llm = self._llm.llm.with_structured_output(MedicalRecordStructure)
            structured_medical_record = structured_llm.invoke([
                {"role": "system", "content": struct_system_prompt},
                {"role": "user", "content": struct_user_prompt}
            ])

            latency = int((time.time() - start_time) * 1000)
            return MedicalRecordOutput(
                ok=True,
                data={
                    "type": "patient_education_parse",
                    "parsed_content": response,
                    "structured_medical_record": structured_medical_record
                },
                latency_ms=latency
            ).model_dump()

        except Exception as e:
            latency = int((time.time() - start_time) * 1000)
            return MedicalRecordOutput(
                ok=False,
                error=str(e),
                latency_ms=latency
            ).model_dump()


def create_medical_record_parser_tool(**kwargs) -> MedicalRecordParserTool:
    return MedicalRecordParserTool(**kwargs)