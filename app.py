import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from pathlib import Path
import json

import numpy as np
import pandas as pd
import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from threading import Thread
import duckdb

# PyLate (Late Interaction: ColBERT + PLAID index)
from pylate import indexes, models, retrieve

# ===== Settings =====
EMBED_MODEL_NAME = os.environ.get("EMBED_MODEL_NAME", "LiquidAI/LFM2-ColBERT-350M")
HF_CHAT_MODEL = os.environ.get("HF_CHAT_MODEL", "LiquidAI/LFM2-1.2B-RAG")
TOP_K = int(os.environ.get("TOP_K", "20"))
SERVER_PORT = int(os.environ.get("GRADIO_SERVER_PORT", os.environ.get("PORT", "7860")))

# PyLate Index settings
INDEX_FOLDER = os.environ.get("PYLATE_INDEX_FOLDER", "pylate-index")
INDEX_NAME = os.environ.get("PYLATE_INDEX_NAME", "index")
ENCODE_BATCH = int(os.environ.get("PYLATE_ENCODE_BATCH", "32"))

INPUT_CSV = Path("input.csv")

# ===== Device selection =====
if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"
DTYPE_GEN = torch.float16 if DEVICE in ("cuda", "mps") else torch.float32

# ===== PyLate（ColBERT）Retriever =====
colbert_model = models.ColBERT(model_name_or_path=EMBED_MODEL_NAME)
# Some tokenizers may miss pad_token; set it to eos_token if available
try:
    if getattr(colbert_model.tokenizer, "pad_token", None) is None and getattr(colbert_model.tokenizer, "eos_token", None) is not None:
        colbert_model.tokenizer.pad_token = colbert_model.tokenizer.eos_token
except Exception:
    pass

def _get_plaid_index(override: bool = False):
    """
    Return PyLate PLAID index. override=True recreates it.
    """
    return indexes.PLAID(
        index_folder=INDEX_FOLDER,
        index_name=INDEX_NAME,
        override=override,
    )

def _id2text_path() -> Path:
    return Path(INDEX_FOLDER) / "id2text.json"

def _load_id2text() -> dict[str, str]:
    p = _id2text_path()
    if p.exists():
        try:
            with p.open("r", encoding="utf-8") as f:
                m = json.load(f)
            # ensure str->str
            return {str(k): str(v) for k, v in m.items()}
        except Exception:
            return {}
    return {}

def _save_id2text(mapping: dict[str, str]) -> None:
    Path(INDEX_FOLDER).mkdir(parents=True, exist_ok=True)
    p = _id2text_path()
    with p.open("w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False)

def get_duckdb_con(client_id, client_secret, oauth_uri, catalog_name, endpoint, s3_region):
    con = duckdb.connect()
    con.execute("INSTALL iceberg;")
    con.execute("LOAD iceberg;")
    con.execute("INSTALL httpfs;")
    con.execute("LOAD httpfs;")
    
    con.execute(f"""
    CREATE SECRET polaris_secret (
        TYPE iceberg,
        CLIENT_ID '{client_id}',
        CLIENT_SECRET '{client_secret}',
        OAUTH2_SERVER_URI '{oauth_uri}'
    );
    """)
    con.execute(f"""
    ATTACH '{catalog_name}' AS mdls_s3 (
            TYPE iceberg,
            SECRET polaris_secret,
            ENDPOINT '{endpoint}',
            DEFAULT_REGION '{s3_region}'
    );
    """)
    con.execute(f"SET s3_region='{s3_region}';")
    return con

def load_iceberg_preview(client_id, client_secret, oauth_uri, catalog_name, endpoint, s3_region, table_name):
    try:
        con = get_duckdb_con(client_id, client_secret, oauth_uri, catalog_name, endpoint, s3_region)
        df = con.execute(f"SELECT * FROM {table_name} LIMIT 10;").df()
        return df, "プレビューを読み込みました。"
    except Exception as e:
        return pd.DataFrame(), f"プレビュー読み込みエラー: {e}"

def rebuild_pylate_index_from_iceberg(client_id, client_secret, oauth_uri, catalog_name, endpoint, s3_region, table_name, id_col, text_col):
    if not client_id or not client_secret:
         return "エラー: Client ID と Client Secret を入力してください。"
    try:
        con = get_duckdb_con(client_id, client_secret, oauth_uri, catalog_name, endpoint, s3_region)
        df = con.execute(f"SELECT {id_col} AS id, {text_col} AS text FROM {table_name};").df()
    except Exception as e:
        return f"Icebergからのデータ読み込みエラー: {e}"

    df = df.dropna(subset=["id", "text"])
    try:
        # id列を文字列として取得
        try:
            ids = [str(int(i)) for i in df["id"].tolist()]
        except Exception:
            ids = [str(i) for i in df["id"].tolist()]
    except Exception:
        return "エラー: 'id' 列のパースに失敗しました。"
        
    documents = df["text"].astype(str).tolist()

    if len(ids) == 0:
        return "エラー: 有効な行がありません（空のテーブル、または指定したカラムが不正）。"

    try:
        index = _get_plaid_index(override=True)
        documents_embeddings = colbert_model.encode(
            documents,
            batch_size=ENCODE_BATCH,
            is_query=False,
            show_progress_bar=True,
        )
        index.add_documents(
            documents_ids=ids,
            documents_embeddings=documents_embeddings,
        )
        _save_id2text({i: t for i, t in zip(ids, documents)})
        return f"PyLateインデックス再構築完了: {len(documents)} 文書 -> {INDEX_FOLDER}/{INDEX_NAME}"
    except Exception as e:
        return f"インデックス構築中にエラー: {e}"

def _fetch_texts_by_ids(ids: list[str]) -> list[str]:
    """
    Map retrieved ids to texts using sidecar mapping under pylate-index/id2text.json.
    """
    if not ids:
        return []
    mapping = _load_id2text()
    return [mapping.get(str(i), "") for i in ids]

# ===== Generator (HF) =====
chat_tokenizer = None
chat_model = None

def ensure_chat_model(consent_download: bool) -> str | None:
    """
    Prepare HF generator model.
    - Try local only (local_files_only=True)
    - If not found and consent_download=True, fetch from network
    - Return error string on failure, or None on success
    """
    global chat_tokenizer, chat_model
    if chat_model is not None and chat_tokenizer is not None:
        return None

    try:
        chat_tokenizer = AutoTokenizer.from_pretrained(HF_CHAT_MODEL, trust_remote_code=True, local_files_only=True)
        chat_model = AutoModelForCausalLM.from_pretrained(
            HF_CHAT_MODEL,
            trust_remote_code=True,
            local_files_only=True,
            dtype=DTYPE_GEN,
        )
        chat_model.to(DEVICE).eval()
        try:
            if getattr(chat_tokenizer, "pad_token_id", None) is None and getattr(chat_tokenizer, "eos_token_id", None) is not None:
                chat_tokenizer.pad_token = chat_tokenizer.eos_token
        except Exception:
            pass
        try:
            if getattr(chat_model, "generation_config", None) is not None and getattr(chat_model.generation_config, "pad_token_id", None) is None and getattr(chat_tokenizer, "eos_token_id", None) is not None:
                chat_model.generation_config.pad_token_id = chat_tokenizer.eos_token_id
        except Exception:
            pass
        return None
    except Exception as e_local:
        if not consent_download:
            return (
                "生成用のHFモデルがローカルに見つかりません。画面の「未ダウンロードならHugging Faceからモデルをダウンロードしてよい」にチェックを入れて再実行してください。\n"
                f"モデル名: {HF_CHAT_MODEL}\n詳細: {e_local}"
            )
        try:
            chat_tokenizer = AutoTokenizer.from_pretrained(HF_CHAT_MODEL, trust_remote_code=True)
            chat_model = AutoModelForCausalLM.from_pretrained(
                HF_CHAT_MODEL,
                trust_remote_code=True,
                dtype=DTYPE_GEN,
            )
            chat_model.to(DEVICE).eval()
            try:
                if getattr(chat_tokenizer, "pad_token_id", None) is None and getattr(chat_tokenizer, "eos_token_id", None) is not None:
                    chat_tokenizer.pad_token = chat_tokenizer.eos_token
            except Exception:
                pass
            try:
                if getattr(chat_model, "generation_config", None) is not None and getattr(chat_model.generation_config, "pad_token_id", None) is None and getattr(chat_tokenizer, "eos_token_id", None) is not None:
                    chat_model.generation_config.pad_token_id = chat_tokenizer.eos_token_id
            except Exception:
                pass
            return None
        except Exception as e_dl:
            return f"Hugging Faceからのモデルダウンロード/ロードに失敗しました: {e_dl}"

def build_prompt(messages: list[dict]) -> str:
    """
    Fallback simple prompt build if no chat template.
    """
    system = ""
    user = ""
    if messages:
        if messages[0].get("role") == "system":
            system = messages[0].get("content", "")
        user = messages[-1].get("content", "")
    prompt = f"{system}\n\nUser: {user}\nAssistant:"
    return prompt


def render_chat_prompt(messages: list[dict]) -> str:
    """
    Render messages using the model's chat template if available, otherwise fallback.
    """
    try:
        return chat_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        return build_prompt(messages)

def generate_with_hf_chat(messages: list[dict]) -> str:
    """
    Generate using tokenizer.apply_chat_template() if available.
    Falls back to plain prompt tokenization on failure.
    """
    try:
        model_inputs = chat_tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(DEVICE)
    except Exception:
        prompt = build_prompt(messages)
        model_inputs = chat_tokenizer(prompt, return_tensors="pt").to(DEVICE)

    # Normalize inputs to a mapping that can be expanded with ** for generate()
    if isinstance(model_inputs, torch.Tensor):
        inputs = {"input_ids": model_inputs}
    else:
        # BatchEncoding or dict-like
        inputs = model_inputs

    # Ensure attention_mask is present to avoid pad/eos ambiguity warnings
    if "attention_mask" not in inputs:
        try:
            inputs["attention_mask"] = torch.ones_like(inputs["input_ids"], dtype=torch.long, device=inputs["input_ids"].device)
        except Exception:
            inputs["attention_mask"] = torch.ones_like(inputs["input_ids"], dtype=torch.long)

    # Collect reasonable EOS token ids (model-specific special tokens included if present)
    eos_ids = set()
    if getattr(chat_tokenizer, "eos_token_id", None) is not None:
        eos_ids.add(chat_tokenizer.eos_token_id)
    for tok in ("<|eot_id|>", "<|im_end|>", "<|end|>"):
        try:
            tid = chat_tokenizer.convert_tokens_to_ids(tok)
            if isinstance(tid, int) and tid >= 0:
                eos_ids.add(tid)
        except Exception:
            pass

    gen_kwargs = dict(
        max_new_tokens=512,
        do_sample=False,
        pad_token_id=chat_tokenizer.eos_token_id if getattr(chat_tokenizer, "eos_token_id", None) is not None else None,
    )
    if eos_ids:
        gen_kwargs["eos_token_id"] = list(eos_ids)

    with torch.no_grad():
        output_ids = chat_model.generate(**inputs, **gen_kwargs)

    input_len = inputs["input_ids"].shape[-1]
    gen_ids = output_ids[0][input_len:]
    text = chat_tokenizer.decode(gen_ids, skip_special_tokens=True)
    return text.strip()

def stream_generate_with_hf_chat(messages: list[dict]):
    """
    Stream tokens using TextIteratorStreamer. Yields incremental decoded text.
    """
    try:
        model_inputs = chat_tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(DEVICE)
    except Exception:
        prompt = build_prompt(messages)
        model_inputs = chat_tokenizer(prompt, return_tensors="pt").to(DEVICE)

    # Normalize inputs into mapping for generate()
    if isinstance(model_inputs, torch.Tensor):
        inputs = {"input_ids": model_inputs}
    else:
        inputs = model_inputs

    # Ensure attention_mask is present to avoid pad/eos ambiguity warnings
    if "attention_mask" not in inputs:
        try:
            inputs["attention_mask"] = torch.ones_like(inputs["input_ids"], dtype=torch.long, device=inputs["input_ids"].device)
        except Exception:
            inputs["attention_mask"] = torch.ones_like(inputs["input_ids"], dtype=torch.long)

    # EOS/PAD setup
    eos_ids = set()
    if getattr(chat_tokenizer, "eos_token_id", None) is not None:
        eos_ids.add(chat_tokenizer.eos_token_id)
    for tok in ("<|eot_id|>", "<|im_end|>", "<|end|>"):
        try:
            tid = chat_tokenizer.convert_tokens_to_ids(tok)
            if isinstance(tid, int) and tid >= 0:
                eos_ids.add(tid)
        except Exception:
            pass

    gen_kwargs = dict(
        max_new_tokens=512,
        do_sample=False,
        pad_token_id=chat_tokenizer.eos_token_id if getattr(chat_tokenizer, "eos_token_id", None) is not None else None,
    )
    if eos_ids:
        gen_kwargs["eos_token_id"] = list(eos_ids)

    # Create streamer and launch generation in background
    streamer = TextIteratorStreamer(chat_tokenizer, skip_prompt=True, skip_special_tokens=True)
    gen_kwargs = {**inputs, **gen_kwargs, "streamer": streamer}
    thread = Thread(target=chat_model.generate, kwargs=gen_kwargs)
    thread.start()

    accumulated = ""
    for piece in streamer:
        accumulated += piece
        yield accumulated

def _build_messages(context: str, user_query: str):
    sys = (
        "You are a helpful assistant. Use the provided context to answer the user's question. "
        "If the answer cannot be found in the context, say you do not know."
    )
    user = (
        f"Context:\n{context}\n\n"
        f"Question: {user_query}\n"
        f"Answer in Japanese if the user asked in Japanese."
    )
    return [
        {"role": "system", "content": sys},
        {"role": "user", "content": user},
    ]

# ===== Retrieval（PyLate index） =====
def _retrieve_with_pylate(query: str, top_k: int = TOP_K):
    """
    Retrieve top-k document texts and scores using PyLate (ColBERT + PLAID index).
    """
    try:
        index = _get_plaid_index(override=False)
        retriever = retrieve.ColBERT(index=index)
    except Exception as e:
        raise RuntimeError(f"インデックスの読み込みに失敗しました。先にIceberg Data Prepタブから『PyLate インデックスを再構築』を実行してください。詳細: {e}")

    try:
        queries_embeddings = colbert_model.encode(
            [query],
            batch_size=ENCODE_BATCH,
            is_query=True,
            show_progress_bar=False,
        )
        scores_all = retriever.retrieve(
            queries_embeddings=queries_embeddings,
            k=int(top_k),
        )
        results = scores_all[0] if scores_all else []
        ids = [str(r["id"]) for r in results]
        scores = np.asarray([float(r["score"]) for r in results], dtype=np.float32)
        texts = _fetch_texts_by_ids(ids)
        return texts, scores
    except Exception as e:
        raise RuntimeError(f"検索中にエラーが発生しました: {e}")

# ===== RAG infer =====
def rag_infer(user_query: str, top_k: int, consent_download: bool):
    user_query = (user_query or "").strip()
    if not user_query:
        return "質問を入力してください。", ""

    try:
        retrieved_texts, top_scores = _retrieve_with_pylate(user_query, top_k=int(top_k))
    except Exception as e:
        return (
            "検索中にエラーが発生しました。Iceberg Data Prepタブから"
            "『PyLate インデックスを再構築』を実行してください。\n"
            f"詳細: {e}",
            ""
        )

    context = "\n\n".join(retrieved_texts)
    messages = _build_messages(context, user_query)

    # Ensure model/tokenizer are available
    err = ensure_chat_model(consent_download=consent_download)

    # Prepare logs (render prompt via chat template if tokenizer is available)
    try:
        scores_list = np.round(np.asarray(top_scores), 4).tolist()
    except Exception:
        scores_list = []
    if chat_tokenizer is not None:
        try:
            rendered_prompt = render_chat_prompt(messages)
        except Exception:
            rendered_prompt = build_prompt(messages)
    else:
        rendered_prompt = build_prompt(messages)

    log_lines = [
        f"Device: {DEVICE}, dtype: {DTYPE_GEN}",
        f"Retriever (PyLate): {EMBED_MODEL_NAME} with PLAID index",
        f"Generator: {HF_CHAT_MODEL}",
        f"TopK: {int(top_k)}",
        f"Query: {user_query}",
        f"Scores: {scores_list}",
        "Retrieved texts:",
    ]
    for i, txt in enumerate(retrieved_texts, 1):
        log_lines.append(f"{i}) {txt}")
    log_lines.append("Rendered prompt (via chat template if available):")
    log_lines.append(rendered_prompt)
    logs = "\n".join(log_lines)

    if err:
        # Stream-compatible early exit
        yield err, logs
        return

    # Stream tokens using tokenizer.apply_chat_template()-based inputs
    try:
        for partial in stream_generate_with_hf_chat(messages):
            yield partial, logs
    except Exception as e_gen:
        yield f"生成中にエラーが発生しました: {e_gen}", logs

# ===== Data Prep (Iceberg/Polaris) =====

# ===== Gradio app =====
with gr.Blocks(title="ローカルRAGデモ（PyLate + HF Transformers）") as demo:
    with gr.Tabs():
        with gr.Tab("Iceberg Data Prep"):
            gr.Markdown(
                """
                ## Polaris/Iceberg連携
                - PolarisカタログおよびIcebergテーブルに接続し、RAG用のデータをロードしてPyLateインデックスを構築します。
                - 対象テーブルには必ず一意なID列と、テキストを格納した列が必要です。
                """
            )
            with gr.Row():
                with gr.Column():
                    client_id = gr.Textbox(label="Client ID")
                    client_secret = gr.Textbox(label="Client Secret", type="password")
                    oauth_uri = gr.Textbox(label="OAuth2 Server URI", value="https://alumni-glowworm.ap-southeast-2.aws.polaris.fivetran.com/api/catalog/v1/oauth/tokens")
                    catalog_name = gr.Textbox(label="Catalog Name", value="insist_underneath")
                with gr.Column():
                    endpoint = gr.Textbox(label="Endpoint", value="https://alumni-glowworm.ap-southeast-2.aws.polaris.fivetran.com/api/catalog/")
                    s3_region = gr.Textbox(label="S3 Region", value="ap-southeast-2")
                    table_name = gr.Textbox(label="Table Name", value="mdls_s3.mhyugaji_oracle_csg_apac_hyugaji.jaffle_shop_customers")
                    with gr.Row():
                        id_col = gr.Textbox(label="ID Column Name", value="id")
                        text_col = gr.Textbox(label="Text Column Name", value="text")

            with gr.Row():
                preview_btn = gr.Button("Preview (10件ロード)")
                index_btn = gr.Button("PyLate インデックスを再構築")

            prep_status = gr.Textbox(label="ステータス/メッセージ", lines=3)
            df_comp = gr.Dataframe(label="プレビューデータ", interactive=False)

            preview_btn.click(
                load_iceberg_preview, 
                inputs=[client_id, client_secret, oauth_uri, catalog_name, endpoint, s3_region, table_name], 
                outputs=[df_comp, prep_status]
            )
            index_btn.click(
                rebuild_pylate_index_from_iceberg,
                inputs=[client_id, client_secret, oauth_uri, catalog_name, endpoint, s3_region, table_name, id_col, text_col],
                outputs=[prep_status]
            )

        with gr.Tab("RAG"):
            gr.Markdown(
                f"""
                # ローカルRAGデモ（PyLate + HF Transformers）
                - Retriever (Late Interaction): {EMBED_MODEL_NAME} with PyLate PLAID index
                - Generator: {HF_CHAT_MODEL}
                - 生成用モデルはローカルに無い場合、下のチェックボックスで同意した上でHugging Faceからダウンロードして利用します。
                """
            )
            with gr.Row():
                with gr.Column(scale=2):
                    inp = gr.Textbox(label="質問を入力してください", lines=4)
                    consent = gr.Checkbox(
                        label="未ダウンロードならHugging Faceからモデルをダウンロードしてよい（初回のみ大容量DLの可能性あり）",
                        value=False,
                    )
                    with gr.Row():
                        topk = gr.Number(label="TopK (取得件数)", value=TOP_K, precision=0)
                        btn = gr.Button("送信")

                with gr.Column(scale=2):
                    out = gr.Textbox(label="回答", lines=12)
            with gr.Row():
                logs = gr.Textbox(label="内部ログ（取得コンテキスト/スコア/プロンプト）", lines=14)
            btn.click(rag_infer, inputs=[inp, topk, consent], outputs=[out, logs])


if __name__ == "__main__":
    # Allow overriding port via GRADIO_SERVER_PORT/PORT env
    demo.launch(server_port=SERVER_PORT)
