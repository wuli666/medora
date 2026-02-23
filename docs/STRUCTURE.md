# medgemma_afu

## 项目结构（精简）

```text
medgemma_afu/
├── api/                 # FastAPI 接口层
├── src/                 # 后端核心逻辑
│   ├── config/          # 配置与日志初始化
│   ├── graph/           # 工作流图构建、节点与状态
│   │   ├── builder.py   # 构建并编排图结构
│   │   ├── nodes.py     # 各业务节点实现
│   │   └── state.py     # 图运行状态定义
│   ├── llm/             # 模型封装与实例工厂
│   ├── prompts/         # Prompt 管理与模板文件
│   │   ├── prompts.py   # Prompt 加载与分发逻辑
│   │   └── templates/   # 各节点使用的提示词模板
│   ├── tool/            # 外部工具能力（检索、PDF、患者库等）
│   │   ├── medgemma_tool.py # MedGemma 工具调用封装
│   │   ├── search_tools.py  # 搜索与检索工具
│   │   ├── pdf_parser.py    # PDF 解析工具
│   │   └── patient_db.py    # 患者数据查询工具
│   ├── runtime/         # 运行时辅助能力（如进度上报）
│   └── utils/           # 通用工具函数
├── frontend/            # 前端工程（Vite + React + UI 组件）
├── tests/               # 后端单元测试/集成测试
├── data/                # 本地数据文件（SQLite/向量库）
├── docs/                # 项目文档与静态图
└── requirements.txt     # Python 依赖
```
