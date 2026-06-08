# ローカルRAGデモ（PyLate + Transformers + Gradio + Iceberg/Polaris）

[English](README_en.md) | [日本語](README.md)

<img width="800" alt="スクリーンショット 2025-11-13 3 03 17" src="https://github.com/user-attachments/assets/b6920340-7c8e-4527-9296-6c321ed749a9" />

## 概要
- PyLate（ColBERT Late Interaction）で高精度なトークン単位検索を行い、取得した文脈を HF Transformers のチャットモデルに与えて回答を生成するローカル RAG デモです。
- データソースとして **DuckDB を介した Polaris カタログ / Iceberg テーブル** と連携し、リモートのデータレイクから直接データをロードしてインデックス化します。
- 検索用インデックスはローカルディレクトリ `pylate-index/` に保存されます（.gitignore 済み）。
- 既定のモデル:
  - Retriever: LiquidAI/LFM2-ColBERT-350M
  - Generator: LiquidAI/LFM2-1.2B-RAG

## クイックスタート

https://www.youtube.com/watch?v=D6Dr2vGgSZw

1) 準備＆起動
- git clone https://github.com/hyugma/sample-rag/
  - ここのリポジトリを自分の環境にコピー （もしくは↑のYoutubeのようにZipとしてダウンロード）
- curl -LsSf https://astral.sh/uv/install.sh | sh
  - uv コマンドのインストール（なければ） 
  - Windowsへのインストールは以下を参照
  - https://docs.astral.sh/uv/getting-started/installation/
- uv sync
  - 必要なモジュールのインストール （DuckDB等を含みます）
- uv run app.py
  - ポートは 7860（GRADIO_SERVER_PORT/PORT で上書き可）
  - 初回起動時はembeddingモデルをDLするので１分くらい待つ可能性あり。

2) 使い方（Gradio UI）
- Iceberg Data Prep タブからデータ投入:
  - Polaris カタログの認証情報（Client ID, Client Secret など）を入力します。
  - 対象のカタログ、エンドポイント、テーブル名、対象の列名（ID列、テキスト列）を指定します。
  - 「Preview (10件ロード)」で接続と内容の確認を行うことができます。
  - 「PyLate インデックスを再構築」をクリックして Iceberg テーブルから全件ロードし、インデックス生成を行います。
- RAG タブ:
  - 初回は生成モデルのダウンロードに同意（チェックボックス）するとダウンロードする
  - 質問と TopK を指定して送信する
  - 取得文脈・スコア・最終プロンプト（テンプレート適用済み）がログに表示される

## 内部処理フロー（ハイレベル）

1) データ管理
- DuckDB を利用して Polaris カタログ・Iceberg テーブルに接続し、指定されたテーブルのデータを DataFrame として取得します。
- 「インデックス再構築」で PLAID 形式のインデックスを `pylate-index/` に生成します。
- id → 原文のマッピングを `id2text.json` に保存します。

2) 検索（Late Interaction）
- クエリを ColBERT エンコーダでベクトル化（is_query=True）
- PLAID インデックスから TopK を取得
- 取得 id を `id2text.json` で本文に引き当て

3) プロンプト組み立て
- 取得した文脈とユーザー質問から chat メッセージ配列を構築（system/user）
- tokenizer.apply_chat_template(messages, add_generation_prompt=True) でモデル固有のチャットテンプレートを適用
- ログには「Rendered prompt」としてテンプレート適用後の実体を出力

4) 生成
- apply_chat_template(..., tokenize=True, return_tensors="pt") でトークナイズした入力を model.generate に渡す
- 適切な EOS/PAD 設定で停止制御
- 生成テキストを decode して回答として返却

## 環境変数（主なもの）
- EMBED_MODEL_NAME（既定: LiquidAI/LFM2-ColBERT-350M）
- HF_CHAT_MODEL（既定: LiquidAI/LFM2-1.2B-RAG）
- TOP_K（既定: 20）
- GRADIO_SERVER_PORT または PORT（既定: 7860）
- PYLATE_INDEX_FOLDER（既定: pylate-index）
- PYLATE_INDEX_NAME（既定: index）

## リポジトリ運用メモ
- pylate-index/、大きなモデルファイル（*.safetensors など）は .gitignore 済み
- 依存は pyproject.toml（uv での運用を想定）

## クレジット
- Liquid AI: https://www.liquid.ai/ja
- LFM2-ColBERT-350M: https://huggingface.co/LiquidAI/LFM2-ColBERT-350M
- LFM2-1.2B-RAG: https://huggingface.co/LiquidAI/LFM2-1.2B-RAG
- PyLate: https://github.com/lightonai/pylate
- DuckDB: https://duckdb.org/

## ライセンス
- 煮るなり焼くなりしてお片付けはご自分で！
