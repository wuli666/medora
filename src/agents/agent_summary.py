import sqlite3
import json
from typing import Dict, Any
from datetime import datetime
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from .output_models import SummaryOutput

DB_PATH = "medical_agent.db"

def store_summary_to_db(session_id: str, summary_struct: SummaryOutput, final_answer: str):
    """
    实际执行 SQLite 写入操作的函数。
    """

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS medical_history (
            session_id TEXT PRIMARY KEY,
            summary_json TEXT,
            created_at TEXT
        )
    ''')

    summary_json_str = summary_struct.model_dump_json()

    cursor.execute('''
        INSERT OR REPLACE INTO medical_history 
        (session_id, summary_json, created_at)
        VALUES (?, ?, ?)
    ''', (
        session_id,
        summary_json_str,
        datetime.utcnow().isoformat()
    ))

    conn.commit()
    conn.close()

def summary_node(state: Dict[str, Any], config: RunnableConfig) -> Dict[str, Any]:
    """
    LangGraph summary node.
    Responsibilities:
    1. 从 state 读取数据
    2. 生成自然语言回答
    3. 生成结构化数据并写入 SQLite
    """

    parsed_record = state.get("parsed_medical_record", {})
    classification = state.get("classification", "unknown")
    raw_input = state.get("raw_input", "")
    session_id = state.get("session_id", "default_session")

    summary_struct = SummaryOutput(
        diagnoses=parsed_record.get("diagnoses", []),
        medications=parsed_record.get("medications", []),
        risk_flags=parsed_record.get("risk_flags", [])
    )

    final_answer_lines = ["我已经帮你整理了这份病历的关键信息：\n"]

    if summary_struct.diagnoses:
        final_answer_lines.append("【医生提到的诊断或考虑】")
        for d in summary_struct.diagnoses:
            final_answer_lines.append(f"- {d}")

    if summary_struct.medications:
        final_answer_lines.append("\n【用药情况】")
        for m in summary_struct.medications:
            final_answer_lines.append(f"- {m}")

    if summary_struct.risk_flags:
        final_answer_lines.append("\n【风险提醒】")
        for r in summary_struct.risk_flags:
            final_answer_lines.append(f"- {r}")

    final_answer_lines.append("\n⚠️ 以上内容是对病历的整理与解释，不替代医生的专业判断。")
    final_answer = "\n".join(final_answer_lines)

    try:
        store_summary_to_db(session_id, summary_struct, final_answer)
    except Exception as e:
        print(f"Error saving to database: {e}")

    return {
        "messages": [AIMessage(content=final_answer)],
        "summary_struct": summary_struct,
    }