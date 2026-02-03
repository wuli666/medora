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

    def _extract_structured_medical_record(self, medical_record: str) -> Dict[str, Any]:
        """
        Extract structured medical record data by calling MedGemma model.
        This replaces the regex-based extraction with LLM-powered structured extraction.
        """
        try:
            # 构建专门用于结构化提取的提示
            struct_system_prompt = (
                "You are a medical information extraction expert. "
                "Your task is to analyze medical records and extract structured information in JSON format.\n\n"
                "Extract the following information from the medical record:\n"
                "1. symptoms: List of patient symptoms mentioned\n"
                "2. diagnoses: List of medical diagnoses or conditions\n"
                "3. medications: List of prescribed medications and dosages\n"
                "4. tests: List of medical tests and results\n"
                "5. uncertainties: List of unclear or missing information\n"
                "6. risk_flags: List of risk factors or urgent concerns\n\n"
                "Return ONLY valid JSON format with these exact keys. "
                "If any category has no information, return an empty list for that key."
            )
            
            struct_user_prompt = f"""
<MEDICAL_RECORD>
{medical_record}
</MEDICAL_RECORD>

Please extract structured medical information from the above record and return it in JSON format.
Use the following structure:
{{
  "symptoms": ["list", "of", "symptoms"],
  "diagnoses": ["list", "of", "diagnoses"],
  "medications": ["list", "of", "medications"],
  "tests": ["list", "of", "tests"],
  "uncertainties": ["list", "of", "uncertainties"],
  "risk_flags": ["list", "of", "risk_flags"]
}}

Return ONLY the JSON object, no additional text or explanation.
"""
            
            # 调用 MedGemma 模型进行结构化提取
            structured_response = self._llm.invoke(
                prompt=struct_user_prompt,
                system_prompt=struct_system_prompt,
            )
            
            # 解析 JSON 响应
            import json
            import re
            
            # 清理响应文本，提取 JSON 部分
            # 移除可能的 Markdown 代码块标记
            cleaned_response = re.sub(r'^```(?:json)?\s*', '', structured_response.strip())
            cleaned_response = re.sub(r'\s*```$', '', cleaned_response)
            
            try:
                structured_data = json.loads(cleaned_response)
                
                # 验证必需的键是否存在
                required_keys = ["symptoms", "diagnoses", "medications", "tests", "uncertainties", "risk_flags"]
                for key in required_keys:
                    if key not in structured_data:
                        structured_data[key] = []
                    # 确保所有值都是列表
                    if not isinstance(structured_data[key], list):
                        structured_data[key] = []
                        
                return structured_data
                
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse structured JSON response: {e}. Response: {structured_response[:200]}...")
                # 如果 JSON 解析失败，返回空结构
                return {
                    "symptoms": [],
                    "diagnoses": [],
                    "medications": [],
                    "tests": [],
                    "uncertainties": [],
                    "risk_flags": []
                }
                
        except Exception as e:
            logger.error(f"Error in structured medical record extraction: {e}")
            # 出错时返回空结构
            return {
                "symptoms": [],
                "diagnoses": [],
                "medications": [],
                "tests": [],
                "uncertainties": [],
                "risk_flags": []
            }

    def _get_prompts(self, medical_record: str):

        system_prompt = (
            "You are a medical educator helping patients understand their medical records.\n"
            "Your task is to translate medical jargon into simple, patient-friendly language.\n\n"
            "Rules:\n"
            "- Explain medical terms in everyday language\n"
            "- Describe what test results mean for the patient\n"
            "- Explain treatment options clearly\n"
            "- Highlight important information patients need to know\n"
            "- Be empathetic and supportive in your explanations\n"
            "- Do NOT provide medical advice - only explain what's in the record"
        )
        user_prompt = f"""
<MEDICAL_RECORD>
{medical_record}
</MEDICAL_RECORD>

Please translate this medical record into patient-friendly language and explain:
1. What medical conditions or diagnoses are mentioned?
2. What do the test results mean?
3. What treatments or medications are prescribed?
4. What should the patient pay attention to?
5. What follow-up care is needed?

Structure your response in clear sections with simple explanations.
"""
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

            # Extract structured medical record data directly from original medical record
            structured_medical_record = self._extract_structured_medical_record(medical_record)

            latency = int((time.time() - start_time) * 1000)
            data = {
                "type": "patient_education_parse",
                "raw_text_length": len(medical_record),
                "parsed_content": response,
                "patient_friendly": True,
                "structured_medical_record": structured_medical_record
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

            # Extract structured medical record data directly from original medical record
            structured_medical_record = self._extract_structured_medical_record(medical_record)

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