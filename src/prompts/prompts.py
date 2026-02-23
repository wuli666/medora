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
    "STRUCTURED_EXTRACTION_SYSTEM_PROMPT": "structured_extraction_system.md",
    "STRUCTURED_EXTRACTION_USER_PROMPT_TEMPLATE": "structured_extraction_user.md",
    "PATIENT_EDUCATION_SYSTEM_PROMPT": "patient_education_system.md",
    "PATIENT_EDUCATION_USER_PROMPT_TEMPLATE": "patient_education_user.md",
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
STRUCTURED_EXTRACTION_SYSTEM_PROMPT = load_prompt_file(
    _PROMPT_FILES["STRUCTURED_EXTRACTION_SYSTEM_PROMPT"]
)
STRUCTURED_EXTRACTION_USER_PROMPT_TEMPLATE = load_prompt_file(
    _PROMPT_FILES["STRUCTURED_EXTRACTION_USER_PROMPT_TEMPLATE"]
)
PATIENT_EDUCATION_SYSTEM_PROMPT = load_prompt_file(
    _PROMPT_FILES["PATIENT_EDUCATION_SYSTEM_PROMPT"]
)
PATIENT_EDUCATION_USER_PROMPT_TEMPLATE = load_prompt_file(
    _PROMPT_FILES["PATIENT_EDUCATION_USER_PROMPT_TEMPLATE"]
)

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
    "STRUCTURED_EXTRACTION_SYSTEM_PROMPT",
    "STRUCTURED_EXTRACTION_USER_PROMPT_TEMPLATE",
    "PATIENT_EDUCATION_SYSTEM_PROMPT",
    "PATIENT_EDUCATION_USER_PROMPT_TEMPLATE",
]
