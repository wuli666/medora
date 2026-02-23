<p align="center">
  <img src="logo1.png" alt="Medora" width="640" />
</p>

<h1 align="center">Medora: Medical Parsing & Disease Management Agent</h1>

<p align="center">
    <img src="https://img.shields.io/badge/python-≥3.11-blue?logo=python&logoColor=white" alt="Python">
    <img src="https://img.shields.io/badge/license-MIT-green" alt="License">
    <img src="https://img.shields.io/badge/LangGraph-1.0+-purple?style=flat&logo=langgraph&logoColor=white" alt="LangGraph">
</p>

<p align="center">
    <a href="README.md">English| <a href="README.zh.md">中文</a>
</p>

https://github.com/user-attachments/assets/7c7dc6c9-baf6-4bb8-a335-5f1f880797fa

## 📊 Background

Chronic diseases have become a major global public health burden, accounting for roughly 74% of worldwide deaths from non-communicable diseases. Patients with chronic conditions often require long-term follow-up, repeated examinations, continuous medication, and longitudinal health monitoring, which leads to large, fragmented medical records that need ongoing comparison and analysis. Studies show only about 21.6% of patients have adequate health literacy, which significantly affects medication understanding, adherence, and outcomes. The chronic disease management market is also growing rapidly, estimated to increase from around $6B to over $17B globally.

## 🏗️ System Architecture

<p align="center">
  <img src="arch.png" alt="architecture" width="800">
</p>

## ✨ Key Highlights

- **Intent routing + multi-stage agent orchestration**: A `Supervisor` first classifies queries as medical vs non-medical using LLMs and rules. Non-medical questions receive a direct LLM response; medical queries flow through a staged pipeline `Planner → Tooler → Searcher → Reflector → Summarizer`, avoiding unnecessary heavy processing for simple requests.
- **Multimodal parsing for text, PDF, and images**: Supports plain-text records, PDF reports, and image screenshots. PDFs are parsed with `parse_pdf` to extract text and embedded images; images are analyzed by the MedGemma-style multimodal model (`medgemma_analyze_image`) and combined with clinical context.
- **Retrieval-augmented knowledge**: `web_search` wraps Tavily medical search while `rag_search` queries a local Chroma collection. `Tooler` / `Searcher` run both online and local retrieval, and an LLM produces readable medical background summaries.
- **Planning–reflection closed-loop for disease management**: `Planner` generates an initial TODO-style long-term care plan from raw input, updates it with analysis and retrieval results, `Reflector` checks consistency and safety, and `Summarizer` outputs a patient-friendly summary and long-term recommendations.

## ⚙️ Features

After a patient uploads clinical notes, reports, or images, Medora extracts diagnoses, medications, and key indicators using the MedGemma model and produces structured summaries readable by patients. The system augments explanations with web search results for medical terminology (without replacing clinical diagnosis), stores parsed results for longitudinal comparison, and generates medication reminders and follow-up prompts.

## 🧭 Implementation Overview

1) Agent orchestration & state flow
- Graph orchestration: Built on LangGraph, the directed graph is `supervisor → planner → {tooler, searcher, reflector} → summarize`. After `tooler/searcher/reflector` finish, control returns to `planner` which decides whether to iterate or summarize.
- State modeling: `MedAgentState` holds `raw_text / images / merged_analysis / search_results / plan / reflection / summary` and is incrementally updated by each node.
- Entry routing: `Supervisor` uses `_classify_query_intent` (LLM + keyword rules) to determine medical intent; non-medical chat ends with a friendly reply, while medical queries initialize plan and proceed to `Planner`.

2) Parsing & retrieval tools
- Text & image parsing: `Tooler` calls `medgemma_analyze_text` and `medgemma_analyze_image` (text extraction and image interpretation). When both text and images exist, a `MERGE_PROMPT` combines results via LLM.
- PDF handling: If a PDF is uploaded, `run_multi_agent` runs `parse_pdf` to extract full text and embedded images, appending text to `raw_text` and converting images to base64 for image analysis.
- Retrieval augmentation: `web_search` wraps Tavily API for concise web snippets and `rag_search` queries a local Chroma `medical_knowledge` collection for top-K paragraphs. Results are summarized with `SEARCH_SUMMARY_PROMPT` for planner and reflector use.

3) Planning, reflection, and summarization
- Planner: On first entry, `PLAN_INIT_PROMPT` creates a baseline TODO-style long-term plan from `raw_text`. After receiving `merged_analysis + search_results`, `PLAN_PROMPT` produces an updated plan and marks `plan_updated` in state.
- Reflector: Uses `REFLECT_PROMPT` to audit consistency, safety, and executability of analysis, retrieval, and plan. If models are unavailable, it returns a soft-fail message and does not block the main flow.
- Summarizer: Uses `SUMMARIZE_PROMPT` to merge analysis, retrieval, plan, and reflection into a patient-facing Chinese/English summary; non-medical chat uses `NON_MEDICAL_REPLY_PROMPT`.

4) Runtime & persistence
- Progress management: `runtime/progress.py` provides a six-stage timeline per run and updates via in-memory structures and SSE (`/api/multi-agent/events/{run_id}`) for frontend subscriptions.
- Patient & follow-up data: `utils/db.py` uses `aiosqlite` to initialize `patients / medical_records / follow_up_plans` tables to store patient info, structured records, and follow-up plans.

## 📁 Project Structure

```
medgemma_afu/
├── api/                   # Backend API service
│   ├── main.py            # FastAPI entrypoint
│   └── schemas.py         # Data models
├── src/                   # Core business logic
│   ├── agents/            # Agent implementations
│   ├── graph/             # Workflow orchestration
│   │   ├── builder.py     # Graph builder
│   │   ├── nodes.py       # Node definitions
│   │   └── state.py       # State management
│   ├── llm/               # LLM integrations
│   ├── prompts/           # Prompt templates
│   ├── runtime/           # Runtime utilities
│   ├── tool/              # Tooling
│   └── utils/             # Helpers
├── frontend/              # Frontend app
│   ├── src/
│   │   ├── components/    # UI components
│   │   ├── pages/         # Pages
│   │   └── lib/           # Client utilities
│   └── package.json       # Frontend deps
├── data/                  # Data storage
│   ├── chroma/            # Vector DB
│   └── patients.db        # Patient DB
└── requirements.txt       # Python dependencies
```
