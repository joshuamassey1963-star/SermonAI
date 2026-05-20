import sqlite3
import numpy as np

from fastembed import TextEmbedding
from llama_cpp import Llama

# =========================
# PATHS
# =========================

DB_PATH = "/data/data/com.termux/files/home/SermonAI/data/chunks.db"

VECTORS_PATH = "/data/data/com.termux/files/home/SermonAI/data/embeddings/vectors.npy"

CHUNK_IDS_PATH = "/data/data/com.termux/files/home/SermonAI/data/embeddings/chunk_ids.npy"

MODEL_PATH = "/data/data/com.termux/files/home/SermonAI/models/gemma-3-4b-it-Q4_K_M.gguf"

# =========================
# LOAD EMBEDDING MODEL
# =========================

print("\n[AI] Loading embedding model...\n")

embedding_model = TextEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)

# =========================
# LOAD VECTORS
# =========================

print("[AI] Loading vectors...\n")

vectors = np.load(VECTORS_PATH)

chunk_ids = np.load(CHUNK_IDS_PATH)

print(f"[AI] Loaded {len(vectors)} vectors")

# =========================
# SQLITE
# =========================

conn = sqlite3.connect(
    DB_PATH,
    check_same_thread=False
)

cursor = conn.cursor()

# =========================
# LOAD GEMMA
# =========================

print("\n[AI] Loading Gemma...\n")

llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=4096,
    n_threads=4,
    n_gpu_layers=0,
    verbose=False
)

print("[AI] Gemma loaded.\n")

# =========================
# MODES
# =========================

MODES = {

    "short": {
        "top_k": 2,
        "chunk_size": 700,
        "tokens": 250,
    },

    "medium": {
        "top_k": 5,
        "chunk_size": 1200,
        "tokens": 900,
    },

    "long": {
        "top_k": 5,
        "chunk_size": 900,
        "tokens": 1200,
    }
}

# =========================
# MEMORY
# =========================

conversation_history = ""

# =========================
# COSINE SIMILARITY
# =========================

def cosine_similarity(query_vector, vectors):

    query_norm = np.linalg.norm(query_vector)

    vector_norms = np.linalg.norm(
        vectors,
        axis=1
    )

    similarity = np.dot(
        vectors,
        query_vector
    ) / (
        vector_norms *
        query_norm +
        1e-10
    )

    return similarity

# =========================
# MAIN SEARCH
# =========================

def offline_rag_search(
    query,
    mode="medium"
):

    global conversation_history

    if mode not in MODES:

        mode = "medium"

    config = MODES[mode]

    query_embedding = list(
        embedding_model.embed([query])
    )[0]

    scores = cosine_similarity(
        query_embedding,
        vectors
    )

    top_indices = np.argsort(
        scores
    )[-config["top_k"]:][::-1]

    evidence_blocks = []

    sources = []

    for idx in top_indices:

        chunk_id = int(
            chunk_ids[idx]
        )

        cursor.execute(
            """
            SELECT
                sermon_title,
                sermon_code,
                chunk_text
            FROM chunks
            WHERE id = ?
            """,
            (chunk_id,)
        )

        row = cursor.fetchone()

        if not row:
            continue

        sermon_title, sermon_code, chunk_text = row

        chunk_text = (
            chunk_text
            .replace("\n", " ")
            .strip()
        )[:config["chunk_size"]]

        sources.append({
            "title": sermon_title,
            "code": sermon_code
        })

        evidence = f"""
SOURCE:
{sermon_title}

CODE:
{sermon_code}

QUOTE:
{chunk_text}
"""

        evidence_blocks.append(
            evidence
        )

    evidence_text = "\n\n".join(
        evidence_blocks
    )

    # ANDROID SAFETY LIMIT
    evidence_text = evidence_text[:2500]

    if mode == "short":

        instruction = """
Give a concise doctrinal answer.
"""

    elif mode == "medium":

        instruction = """
Give a detailed doctrinal explanation.
"""

    else:

        instruction = """
Write a deep sermon-style doctrinal teaching.

Use many paragraphs.

Explain deeply.
"""

    prompt = f"""
You are an advanced sermon research assistant.

{instruction}

Use ONLY sermon evidence.

Do NOT invent doctrine.

Conversation History:
{conversation_history}

QUESTION:
{query}

SERMON EVIDENCE:
{evidence_text}

ANSWER:
"""

    output = llm(
        prompt,
        max_tokens=config["tokens"],
        temperature=0.2,
        stop=["</s>"]
    )

    answer = output["choices"][0]["text"]

    conversation_history += f"\nUSER: {query}\nAI: {answer}\n"

    # LIMIT MEMORY SIZE
    conversation_history = conversation_history[-1500:]

    return {
        "answer": answer,
        "sources": sources
    }
