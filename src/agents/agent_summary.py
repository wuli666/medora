import sqlite3
import json
from typing import Dict, Any
from datetime import datetime
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

# 数据库文件路径
DB_PATH = "medical_agent.db"

def store_summary_to_db(session_id: str, summary_struct: Dict[str, Any], final_answer: str):
    """
    实际执行 SQLite 写入操作的函数。
    """
    # 1. 连接数据库（如果不存在则自动创建文件）
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # 2. 创建表结构（如果尚未创建）
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS medical_history (
            session_id TEXT PRIMARY KEY,
            case_type TEXT,
            summary_json TEXT,
            final_answer TEXT,
            created_at TEXT
        )
    ''')

    # 3. 准备数据：将字典转换为 JSON 字符串存储
    summary_json_str = json.dumps(summary_struct, ensure_ascii=False)

    # 4. 执行插入或替换
    cursor.execute('''
        INSERT OR REPLACE INTO medical_history 
        (session_id, case_type, summary_json, final_answer, created_at)
        VALUES (?, ?, ?, ?, ?)
    ''', (
        session_id,
        summary_struct.get("case_type", "unknown"),
        summary_json_str,
        final_answer,
        summary_struct.get("created_at")
    ))

    # 5. 提交并关闭
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

    # ---- 1. 从 state 中取数据 ----
    parsed_record = state.get("parsed_medical_record", {})
    classification = state.get("classification", "unknown")
    raw_input = state.get("raw_input", "")
    session_id = state.get("session_id", "default_session") # 新增：获取会话ID

    # ---- 2. 构造结构化 summary（用于 SQLite）----
    summary_struct = {
        "case_type": classification,
        "key_symptoms": parsed_record.get("symptoms", []),
        "diagnoses": parsed_record.get("diagnoses", []),
        "medications": parsed_record.get("medications", []),
        "tests": parsed_record.get("tests", []),
        "uncertainties": parsed_record.get("uncertainties", []),
        "risk_flags": parsed_record.get("risk_flags", []),
        "created_at": datetime.utcnow().isoformat(),
    }

    # ---- 3. 构造给用户的最终回答（自然语言）----
    final_answer_lines = ["我已经帮你整理了这份病历的关键信息：\n"]

    if summary_struct["key_symptoms"]:
        final_answer_lines.append("【主要症状】")
        for s in summary_struct["key_symptoms"]:
            final_answer_lines.append(f"- {s}")

    if summary_struct["diagnoses"]:
        final_answer_lines.append("\n【医生提到的诊断或考虑】")
        for d in summary_struct["diagnoses"]:
            final_answer_lines.append(f"- {d}")

    # ... (省略中间类似的药物、检查拼接代码以保持简洁) ...

    final_answer_lines.append("\n⚠️ 以上内容是对病历的整理与解释，不替代医生的专业判断。")
    final_answer = "\n".join(final_answer_lines)

    # ---- 4. 关键补充：实际写入数据库 ----
    try:
        store_summary_to_db(session_id, summary_struct, final_answer)
    except Exception as e:
        print(f"Error saving to database: {e}")

    # ---- 5. 返回 state 更新 ----
    return {
        "messages": [AIMessage(content=final_answer)],
        "summary_struct": summary_struct,
    }