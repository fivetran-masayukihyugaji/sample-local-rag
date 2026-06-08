# Local RAG Demo (PyLate + Transformers + Gradio + Iceberg/Polaris)

[English](README_en.md) | [日本語](README.md)

<img width="800" alt="Screenshot 2025-11-13 3 03 17" src="https://github.com/user-attachments/assets/b6920340-7c8e-4527-9296-6c321ed749a9" />

## Overview
- This is a local RAG (Retrieval-Augmented Generation) demo that performs highly accurate token-level retrieval using PyLate (ColBERT Late Interaction), feeds the retrieved context into an HF Transformers chat model, and generates answers.
- As a data source, it integrates with **Polaris Catalogs / Iceberg Tables via DuckDB**, allowing it to load and index data directly from a remote data lake.
- The retrieval index is stored locally in the `pylate-index/` directory (ignored in `.gitignore`).
- Default Models:
  - Retriever: LiquidAI/LFM2-ColBERT-350M
  - Generator: LiquidAI/LFM2-1.2B-RAG

## Quick Start

https://www.youtube.com/watch?v=D6Dr2vGgSZw

1) Setup & Launch
- `git clone https://github.com/hyugma/sample-rag/`
  - Clone this repository to your local environment (or download it as a Zip file as shown in the YouTube video above).
- `curl -LsSf https://astral.sh/uv/install.sh | sh`
  - Install the `uv` command if you don't have it.
  - For Windows installation, refer to:
  - https://docs.astral.sh/uv/getting-started/installation/
- `uv sync`
  - Install the required modules (including DuckDB, etc.).
- `uv run app.py`
  - The default port is 7860 (can be overridden by `GRADIO_SERVER_PORT` or `PORT`).
  - On the first launch, it may take about a minute to download the embedding model.

2) Usage (Gradio UI)
- **Iceberg Data Prep Tab** (Data Loading):
  - Enter the authentication credentials for your Polaris catalog (Client ID, Client Secret, etc.).
  - Specify the target catalog, endpoint, table name, and target column names (ID column, Text column).
  - Click "Preview (load 10 rows)" to test the connection and review the data.
  - Click "Rebuild PyLate Index" to load all the data from the Iceberg table and build the retrieval index.
- **RAG Tab**:
  - The first time you use it, check the box to consent to downloading the generation model.
  - Enter your question, specify the `TopK`, and hit send.
  - The retrieved context, scores, and final prompt (with the template applied) will be displayed in the logs.

## Internal Processing Flow (High-Level)

1) Data Management
- Connects to the Polaris catalog and Iceberg tables using DuckDB and retrieves the specified table data as a DataFrame.
- Clicking "Rebuild Index" generates a PLAID format index in `pylate-index/`.
- Saves the ID to original text mapping in `id2text.json`.

2) Retrieval (Late Interaction)
- Vectorizes the query using the ColBERT encoder (`is_query=True`).
- Retrieves the TopK results from the PLAID index.
- Maps the retrieved IDs back to the original text using `id2text.json`.

3) Prompt Assembly
- Constructs a chat message array (system/user) from the retrieved context and the user's question.
- Applies the model-specific chat template using `tokenizer.apply_chat_template(messages, add_generation_prompt=True)`.
- Outputs the actual prompt after template application to the log as "Rendered prompt".

4) Generation
- Tokenizes the input using `apply_chat_template(..., tokenize=True, return_tensors="pt")` and passes it to `model.generate`.
- Controls stopping using appropriate EOS/PAD settings.
- Decodes the generated text and returns it as the answer.

## Environment Variables (Main)
- `EMBED_MODEL_NAME` (Default: LiquidAI/LFM2-ColBERT-350M)
- `HF_CHAT_MODEL` (Default: LiquidAI/LFM2-1.2B-RAG)
- `TOP_K` (Default: 20)
- `GRADIO_SERVER_PORT` or `PORT` (Default: 7860)
- `PYLATE_INDEX_FOLDER` (Default: pylate-index)
- `PYLATE_INDEX_NAME` (Default: index)

## Repository Operation Notes
- `pylate-index/` and large model files (like `*.safetensors`) are added to `.gitignore`.
- Dependencies are managed in `pyproject.toml` (intended to be run with `uv`).

## Credits
- Liquid AI: https://www.liquid.ai/
- LFM2-ColBERT-350M: https://huggingface.co/LiquidAI/LFM2-ColBERT-350M
- LFM2-1.2B-RAG: https://huggingface.co/LiquidAI/LFM2-1.2B-RAG
- PyLate: https://github.com/lightonai/pylate
- DuckDB: https://duckdb.org/

## License
- Feel free to modify and use it as you like, but please clean up after yourself!
