from langchain_core.messages import HumanMessage

from src.llm.model_factory import get_chat_model
from src.prompts.prompts import MEDGEMMA_IMAGE_PROMPT, MEDGEMMA_TEXT_PROMPT


async def medgemma_analyze_text(medical_text: str) -> str:
    prompt = MEDGEMMA_TEXT_PROMPT.format(medical_text=medical_text)
    llm = get_chat_model("TOOLER_TEXT", default_model="qwen-plus", temperature=0.2)
    if llm is None:
        return "文本分析模型不可用：请检查对应 agent 的模型配置。"
    try:
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        return response.content
    except Exception as e:
        return f"MedGemma 文本分析失败: {e}"


async def medgemma_analyze_image(image_base64: str, clinical_context: str = "") -> str:
    prompt_text = MEDGEMMA_IMAGE_PROMPT.format(
        clinical_context=clinical_context or "无额外临床背景信息"
    )
    llm = get_chat_model("TOOLER_IMAGE", default_model="qwen-plus", temperature=0.2)
    if llm is None:
        return "图像分析模型不可用：请检查对应 agent 的模型配置。"
    try:
        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt_text},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
                },
            ]
        )
        response = await llm.ainvoke([message])
        return response.content
    except Exception as e:
        return f"MedGemma 图像分析失败: {e}"
