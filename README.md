# LlamaIndex RAG Agent Example

[中文](#中文) | [English](#english)

---

<a id="中文"></a>
## 🇨🇳 中文说明

这是一个基于 LlamaIndex 构建的 RAG（检索增强生成）示例项目，演示了如何搭建一个支持 PDF 文档问答的智能助手，并包含完整的自动化评测管线。

### ✨ 核心功能

*   **RAG 问答系统**：
    *   支持 PDF 文档加载与解析
    *   使用 PGVector 进行向量存储与检索
    *   基于 OpenAI 兼容接口（支持 DeepSeek, Moonshot 等）的 LLM 回答
    *   检索失败时的自动拦截机制（避免幻觉）
*   **自动化评测管线**：
    *   自动生成测试数据集（支持指定中文/英文提问）
    *   **异步并发评测**：高效评估大量样本，支持自定义并发度
    *   多维度指标评估：忠实度 (Faithfulness)、相关性 (Relevancy)、正确性 (Correctness)
    *   自动生成评测报告（CSV 数据表 + 箱线图统计）
*   **稳健性设计**：
    *   LLM/Embedding 调用支持自动重试与超时控制
    *   完善的错误处理与日志

### 🛠️ 环境依赖

*   Python >= 3.12
*   PostgreSQL + pgvector 插件
*   OpenAI 兼容的 LLM API 服务

### 🚀 快速开始

1.  **安装依赖**
    ```bash
    uv sync
    ```

2.  **配置环境变量**
    复制 `.env.example` 为 `.env` 并填入您的 API Key 和数据库配置：
    ```bash
    cp .env.example .env
    ```
    *关键配置项：*
    *   `LLM_API_KEY`, `LLM_API_BASE`: 模型服务配置
    *   `PGVECTOR_URL`: 向量数据库连接串
    *   `EVAL_QUESTION_LANGUAGE`: 评测问题生成语言 (`zh` 或 `en`)
    *   `EVAL_CONCURRENCY`: 评测并发度（默认 3）

3.  **运行 RAG 问答应用**
    ```bash
    # 首次运行需加 --rebuild 参数以导入数据
    uv run main.py --rebuild
    
    # 后续直接运行
    uv run main.py
    ```

4.  **运行自动化评测**
    ```bash
    # 运行评测（默认使用现有数据集）
    uv run evaluate.py
    
    # 强制重新生成测试数据集
    uv run evaluate.py --rebuild-dataset
    ```

### 📊 评测结果

评测运行完成后，结果将保存在 `eval_outputs/` 目录下：
*   `eval_results.csv`: 包含每个问题的详细评分
*   `eval_summary.txt`: 整体平均分统计
*   `eval_metrics_distribution.png`: 分数分布箱线图

---

<a id="english"></a>
## 🇺🇸 English Description

This is a RAG (Retrieval-Augmented Generation) example project built with LlamaIndex, demonstrating how to build an intelligent assistant for PDF document Q&A, complete with a full automated evaluation pipeline.

### ✨ Key Features

*   **RAG Q&A System**:
    *   PDF document loading and parsing
    *   Vector storage and retrieval using PGVector
    *   LLM response generation via OpenAI-compatible APIs
    *   Automatic fallback mechanism for retrieval failures (prevents hallucinations)
*   **Automated Evaluation Pipeline**:
    *   Auto-generation of test datasets (supports Chinese/English questions)
    *   **Async Concurrent Evaluation**: Efficiently evaluates large samples with customizable concurrency
    *   Multi-dimensional metrics: Faithfulness, Relevancy, Correctness
    *   Auto-generated evaluation reports (CSV data + Boxplot statistics)
*   **Robust Design**:
    *   Automatic retry and timeout control for LLM/Embedding calls
    *   Comprehensive error handling and logging

### 🛠️ Requirements

*   Python >= 3.12
*   PostgreSQL + pgvector extension
*   OpenAI-compatible LLM API service

### 🚀 Quick Start

1.  **Install Dependencies**
    ```bash
    uv sync
    ```

2.  **Configure Environment**
    Copy `.env.example` to `.env` and fill in your API Key and database config:
    ```bash
    cp .env.example .env
    ```
    *Key Configurations:*
    *   `LLM_API_KEY`, `LLM_API_BASE`: Model service configuration
    *   `PGVECTOR_URL`: Vector database connection string
    *   `EVAL_QUESTION_LANGUAGE`: Language for generated evaluation questions (`zh` or `en`)
    *   `EVAL_CONCURRENCY`: Evaluation concurrency level (default: 3)

3.  **Run RAG Application**
    ```bash
    # Use --rebuild flag for the first run to ingest data
    uv run main.py --rebuild
    
    # Subsequent runs
    uv run main.py
    ```

4.  **Run Automated Evaluation**
    ```bash
    # Run evaluation (uses existing dataset by default)
    uv run evaluate.py
    
    # Force regenerate test dataset
    uv run evaluate.py --rebuild-dataset
    ```

### 📊 Evaluation Results

After the evaluation completes, results are saved in the `eval_outputs/` directory:
*   `eval_results.csv`: Detailed scores for each question
*   `eval_summary.txt`: Overall average score statistics
*   `eval_metrics_distribution.png`: Boxplot of score distribution
