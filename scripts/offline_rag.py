import sqlite3
import numpy as np
from fastembed import TextEmbedding
from llama_cpp import Llama

# =========================
# CONFIG
# =========================

DB_PATH = "/data/data/com.termux/files/home/SermonAI/data/chunks.db"

VECTORS_PATH = "/data/data/com.termux/files/home/SermonAI/data/embeddings/vectors.npy"

CHUNK_IDS_PATH = "/data/data/com.termux/files/home/SermonAI/data/embeddings/chunk_ids.npy"

MODEL_PATH = "/data/data/com.termux/files/home/SermonAI/models/gemma-3-4b-it-Q4_K_M.gguf"

# =========================
# LOAD EMBEDDING MODEL
# =========================

print("\nLoading embedding model...\n")

embedding_model = TextEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)

# =========================
# LOAD VECTORS
# =========================

print("Loading vectors...\n")

vectors = np.load(VECTORS_PATH)

chunk_ids = np.load(CHUNK_IDS_PATH)

print(f"Loaded {len(vectors)} vectors")

# =========================
# SQLITE
# =========================

conn = sqlite3.connect(DB_PATH)

cursor = conn.cursor()

# =========================
# LOAD GEMMA
# =========================

print("\nLoading Gemma...\n")

llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=4096,
    n_threads=4,
    n_gpu_layers=0,
    verbose=False
)

print("Gemma loaded.\n")

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
# MODES
# =========================

MODES = {

    "short": {
        "top_k": 2,
        "chunk_size": 700,
        "tokens": 250,
        "instruction": """
Give a concise doctrinal answer.

Use sermon evidence only.

Keep answer short and direct.
"""
    },

    "medium": {
        "top_k": 5,
        "chunk_size": 1200,
        "tokens": 900,
        "instruction": """
Write a doctrinal explanation.

Explain spiritual meaning carefully.

Use sermon evidence properly.

Answer with moderate detail.
"""
    },

    "long": {
        "top_k": 12,
        "chunk_size": 1800,
        "tokens": 2600,
        "instruction": """
Write a very detailed sermon-style doctrinal teaching.

Response must contain at least 10 large paragraphs.

Each paragraph must explain a different aspect.

Explain deeply:

- doctrinal meaning
- spiritual meaning
- symbolic meaning
- prophetic meaning
- historical meaning
- biblical meaning
- revelation meaning
- practical Christian meaning
- related sermon concepts
- connections between doctrines

Use sermon evidence thoroughly.

Do not give short summaries.

Teach slowly and deeply like a preacher teaching doctrine.

Expand ideas carefully.

Doctrinal depth is more important than brevity.
"""
    }
}

# =========================
# MEMORY
# =========================

conversation_history = ""

# =========================
# CHAT LOOP
# =========================

while True:

    print("\nModes:\nshort / medium / long")

    mode = input("\nMode:\n> ").strip().lower()

    if mode not in MODES:

        print("\nInvalid mode.\n")

        continue

    config = MODES[mode]

    query = input("\nAsk:\n> ")

    if query.lower() in ["exit", "quit"]:

        break

    print("\nGenerating embedding...\n")

    query_embedding = list(
        embedding_model.embed([query])
    )[0]

    print("Searching sermons...\n")

    scores = cosine_similarity(
        query_embedding,
        vectors
    )

    top_indices = np.argsort(
        scores
    )[-config["top_k"]:][::-1]

    evidence_blocks = []

    for rank, idx in enumerate(
        top_indices,
        start=1
    ):

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

    prompt = f"""
You are an advanced sermon research assistant.

{config["instruction"]}

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

    print("\nGenerating doctrinal answer...\n")

    output = llm(
        prompt,
        max_tokens=config["tokens"],
        temperature=0.2,
        stop=["</s>"]
    )

    answer = output["choices"][0]["text"]

    print("\n====================================\n")

    print(answer)

    print("\n====================================\n")

    conversation_history += f"\nUSER: {query}\nAI: {answer}\n"

conn.close()
