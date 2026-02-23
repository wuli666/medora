from pathlib import Path

_PROMPTS_DIR = Path(__file__).resolve().parent / "templates"
_PROMPT_FILES: dict[str, str] = {
    "MEDGEMMA_TEXT_PROMPT": "medgemma_text.md",
    "MEDGEMMA_IMAGE_PROMPT": "medgemma_image.md",
    "MERGE_PROMPT": "merge.md",
    "SEARCH_SUMMARY_PROMPT": "search_summary.md",
    "PLAN_PROMPT": "plan.md",
    "PLAN_INIT_PROMPT": "plan_init.md",
    "PLAN_TOOL_PROMPT": "plan_tool.md",
    "REFLECT_PROMPT": "reflect.md",
    "SUMMARIZE_PROMPT": "summarize.md",
    "NON_MEDICAL_REPLY_PROMPT": "non_medical_reply.md",
    "SUPERVISOR_EVAL_PROMPT": "supervisor_eval.md",
    "PLANNER_DECIDE_PROMPT": "planner_decide.md",
    "INTENT_CLASSIFY_PROMPT": "intent_classify.md",
}


def load_prompt_file(filename: str) -> str:
    path = _PROMPTS_DIR / filename
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"Failed to load prompt file: {path}") from exc


MEDGEMMA_TEXT_PROMPT = load_prompt_file(_PROMPT_FILES["MEDGEMMA_TEXT_PROMPT"])
MEDGEMMA_IMAGE_PROMPT = load_prompt_file(_PROMPT_FILES["MEDGEMMA_IMAGE_PROMPT"])
MERGE_PROMPT = load_prompt_file(_PROMPT_FILES["MERGE_PROMPT"])
SEARCH_SUMMARY_PROMPT = load_prompt_file(_PROMPT_FILES["SEARCH_SUMMARY_PROMPT"])
PLAN_PROMPT = load_prompt_file(_PROMPT_FILES["PLAN_PROMPT"])
PLAN_INIT_PROMPT = load_prompt_file(_PROMPT_FILES["PLAN_INIT_PROMPT"])
PLAN_TOOL_PROMPT = load_prompt_file(_PROMPT_FILES["PLAN_TOOL_PROMPT"])
REFLECT_PROMPT = load_prompt_file(_PROMPT_FILES["REFLECT_PROMPT"])
SUMMARIZE_PROMPT = load_prompt_file(_PROMPT_FILES["SUMMARIZE_PROMPT"])
NON_MEDICAL_REPLY_PROMPT = load_prompt_file(_PROMPT_FILES["NON_MEDICAL_REPLY_PROMPT"])
SUPERVISOR_EVAL_PROMPT = load_prompt_file(_PROMPT_FILES["SUPERVISOR_EVAL_PROMPT"])
PLANNER_DECIDE_PROMPT = load_prompt_file(_PROMPT_FILES["PLANNER_DECIDE_PROMPT"])
INTENT_CLASSIFY_PROMPT = load_prompt_file(_PROMPT_FILES["INTENT_CLASSIFY_PROMPT"])


STRUCTURED_EXTRACTION_SYSTEM_PROMPT = """You are a medical information extraction expert. Your task is to analyze medical records and extract structured information in JSON format.

Extract the following information from the medical record:
1. symptoms: List of patient symptoms mentioned
2. diagnoses: List of medical diagnoses or conditions
3. medications: List of prescribed medications and dosages
4. tests: List of medical tests and results
5. uncertainties: List of unclear or missing information
6. risk_flags: List of risk factors or urgent concerns

Return ONLY valid JSON format with these exact keys. If any category has no information, return an empty list for that key."""

STRUCTURED_EXTRACTION_USER_PROMPT_TEMPLATE = """<MEDICAL_RECORD>
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

Return ONLY the JSON object, no additional text or explanation."""

PATIENT_EDUCATION_SYSTEM_PROMPT = """You are a medical educator helping patients understand their medical records.
Your task is to translate medical jargon into simple, patient-friendly language.

Rules:
- Explain medical terms in everyday language
- Describe what test results mean for the patient
- Explain treatment options clearly
- Highlight important information patients need to know
- Be empathetic and supportive in your explanations
- Do NOT provide medical advice - only explain what's in the record"""

PATIENT_EDUCATION_USER_PROMPT_TEMPLATE = """<MEDICAL_RECORD>
{medical_record}
</MEDICAL_RECORD>

Please translate this medical record into patient-friendly language and explain:
1. What medical conditions or diagnoses are mentioned?
2. What do the test results mean?
3. What treatments or medications are prescribed?
4. What should the patient pay attention to?
5. What follow-up care is needed?

Structure your response in clear sections with simple explanations."""

__all__ = [
    "MEDGEMMA_TEXT_PROMPT",
    "MEDGEMMA_IMAGE_PROMPT",
    "MERGE_PROMPT",
    "SEARCH_SUMMARY_PROMPT",
    "PLAN_PROMPT",
    "PLAN_INIT_PROMPT",
    "PLAN_TOOL_PROMPT",
    "REFLECT_PROMPT",
    "SUMMARIZE_PROMPT",
    "NON_MEDICAL_REPLY_PROMPT",
    "SUPERVISOR_EVAL_PROMPT",
    "PLANNER_DECIDE_PROMPT",
    "INTENT_CLASSIFY_PROMPT",
]
