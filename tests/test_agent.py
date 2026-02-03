"""
Test script to demonstrate the refactored MedicalRecordParserTool 
and its integration into the Summary Node.
"""

import json
import time
from typing import Dict, Any
from langchain_core.messages import HumanMessage, AIMessage

# 导入你项目中的模块
from src.tool.medical_record import MedicalRecordParserTool
from src.agents.agent_summary import summary_node 

def test_step_1_medical_tool_execution():
    """测试第一步：调用病历解析工具"""
    print("="*60)
    print("STEP 1: Testing Medical Record Tool")
    print("="*60)

    # 1. 准备测试数据 (使用你提供的 John Doe 案例)
    sample_record = """
    Patient: John Doe, 52-year-old male
    Assessment: Acute inferior STEMI
    Plan: Immediate cardiac catheterization, Aspirin 325mg chewed, 
          Clopidogrel 600mg loading dose, Heparin bolus
    """

    # 2. 初始化工具
    print(f"Initializing MedicalRecordParserTool...")
    tool = MedicalRecordParserTool()

    # 3. 执行工具
    print("Invoking tool (this calls MedGemma via Ollama)...")
    start_time = time.time()
    
    # 注意：根据我们之前的重构，这里返回的是一个 standardized dict
    result = tool._run(medical_record=sample_record)
    
    end_time = time.time()
    print(f"Tool execution completed in {int((end_time - start_time)*1000)}ms.")
    print()

    # 4. 验证标准化输出
    print("Standardized Output Verification:")
    print(f"  - Status (ok): {result.get('ok')}")
    print(f"  - Latency recorded: {result.get('latency_ms')} ms")
    print(f"  - Data Source: {result.get('source')}")
    
    if result["ok"]:
        content = result["data"]["parsed_content"]
        print(f"  - Content Preview: {content}...")
    else:
        print(f"  - Error encountered: {result.get('error')}")

    print()
    return result

def test_step_2_summary_node_integration(tool_output):
    print("="*60)
    print("STEP 2: Testing Summary Node Integration & SQLite Storage")
    print("="*60)

    if not tool_output["ok"]:
        print("Skipping Step 2 because Tool execution failed.")
        return

    # 1. 使用工具返回的真实结构化数据
    structured_medical_record = tool_output["data"].get("structured_medical_record", {})
    
    # 如果没有结构化数据，则使用默认值（向后兼容）
    if not structured_medical_record or all(len(v) == 0 for v in structured_medical_record.values()):
        print("Warning: No structured medical record data found, using default values for testing.")
        structured_medical_record = {
            "symptoms": ["胸痛", "呼吸困难"],
            "diagnoses": ["急性下壁心肌梗死"],
            "medications": ["阿司匹林 325mg", "氯吡格雷 600mg", "肝素"],
            "tests": ["心电图: ST段抬高", "肌钙蛋白 I: 8.2 ng/mL"],
            "uncertainties": ["过敏史不明确"],
            "risk_flags": ["高危：需立即进行心脏导管手术"]
        }

    # 2. 准备符合 summary_node 接口的 state 数据
    summary_state = {
        "raw_input": "请帮我解释这份病历",
        "parsed_medical_record": structured_medical_record,  # 使用真实的结构化数据
        "classification": "medical_record",                  # 对应 summary_struct 中的 case_type
        "messages": [HumanMessage(content="请帮我解释这份病历")],
        "session_id": "test_session_123"                     # 对应 SQLite 中的主键
    }

    print(f"State prepared. Session ID: {summary_state['session_id']}")
    print("Calling actual summary_node (this will now write to SQLite)...")
    
    # 3. 调用实际的 summary_node
    from langchain_core.runnables import RunnableConfig
    config: RunnableConfig = {}
    
    # 此时执行 summary_node 会内部触发 store_summary_to_db
    summary_result = summary_node(summary_state, config)
    
    print("Summary node execution completed.")
    print()
    
    # 4. 显示生成的总结消息
    final_message = summary_result["messages"][0].content
    print("Final Summary Message generated:")
    print("-" * 30)
    print(final_message)
    print("-" * 30)
    
    # 5. 显示结构化数据
    print("\nStructured data for storage (summary_struct):")
    print(json.dumps(summary_result["summary_struct"], indent=2, ensure_ascii=False))
    
    # 6. 验证 SQLite 存储结果
    import os
    import sqlite3
    db_file = "medical_agent.db"
    
    print("\n[Database Verification]")
    if os.path.exists(db_file):
        print(f"  ✓ Database file '{db_file}' exists.")
        
        # 尝试读取刚才存入的数据
        try:
            conn = sqlite3.connect(db_file)
            cursor = conn.cursor()
            cursor.execute("SELECT session_id, created_at FROM medical_history WHERE session_id = ?", 
                           (summary_state["session_id"],))
            row = cursor.fetchone()
            if row:
                print(f"  ✓ Successfully verified record in DB. Session: {row[0]}, Created At: {row[1]}")
            conn.close()
        except Exception as e:
            print(f"  ✗ Failed to query database: {e}")
    else:
        print(f"  ✗ Database file '{db_file}' was NOT created. Please check summary_node logic.")

def main():
    """主测试流程"""
    print("Starting Comprehensive Medical Agent Workflow Test...\n")
    
    # 步骤 1: 运行工具
    tool_result = test_step_1_medical_tool_execution()
    
    # 步骤 2: 运行总结
    test_step_2_summary_node_integration(tool_result)

    print("\n" + "="*60)
    print("WORKFLOW TEST COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()