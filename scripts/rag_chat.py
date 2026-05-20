import sqlite3
import numpy as np
from pathlib import Path
from fastembed import TextEmbedding
import subprocess
import tempfile

BASE_DIR = Path.home() / "SermonAI"

DB_PATH = BASE_DIR / "data" / "chunks.db"

VECTORS_PATH = BASE_DIR / "data" / "embeddings" / "vectors.npy"

CHUNK_IDS_PATH = BASE_DIR / "data" / "embeddings" / "chunk_ids.npy"

MODEL_PATH = BASE_DIR / "models" / "gemma-3-4b-it-Q4_K_M.gguf"

LLAMA_BIN = BASE_DIR / "llama.cpp" / "build" / "bin" / "llama-simple"

TOP_K = 25

MAX_CHUNK = 700

print("Loading embedding model...")

embedding_model = TextEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)

print("Loading vectors...")

vectors = np.load(VECTORS_PATH)

chunk_ids = np.load(CHUNK_IDS_PATH)

print(f"Loaded vectors: {vectors.shape}")

query = input("\nAsk Question: ")

print("\nGenerating query embedding...")

query_embedding = list(
    embedding_model.embed([query])
)[0]

query_embedding = np.array(query_embedding)

print("Searching sermon knowledge...")

similarities = np.dot(vectors, query_embedding)

top_indices = np.argsort(similarities)[-TOP_K:][::-1]

conn = sqlite3.connect(DB_PATH)

cursor = conn.cursor()

doctrine_points = []

sources = []

used_sermons = set()

for idx in top_indices:

    chunk_id = int(chunk_ids[idx])

    cursor.execute("""
        SELECT sermon_title, sermon_code, chunk_text
        FROM chunks
        WHERE id=?
    """, (chunk_id,))

    row = cursor.fetchone()

    if not row:
        continue

    sermon_title, sermon_code, chunk_text = row

    key = f"{sermon_title}_{sermon_code}"

    if key in used_sermons:
        continue

    used_sermons.add(key)

    chunk_text = chunk_text.replace("\n", " ")

    chunk_text = chunk_text[:MAX_CHUNK]

    doctrine_points.append(f"""

SOURCE:
{sermon_title} ({sermon_code})

QUOTATION:
{chunk_text}

""")

    sources.append(
        f"{sermon_title} ({sermon_code})"
    )

conn.close()

context = "\n".join(doctrine_points)

prompt = f"""
You are a William Branham doctrinal research assistant.

You MUST obey these rules:

- Use ONLY the sermon material provided
- Never use outside theology
- Explain doctrine carefully
- Synthesize multiple sermons together
- Give meaningful theological answers
- Quote sermons naturally
- Be detailed and intelligent
- If answer is incomplete, say so
- Do NOT hallucinate

QUESTION:
{query}

SERMON RESEARCH MATERIAL:
{context}

Now produce:

1. A doctrinal explanation
2. Main teaching points
3. Supporting sermon references
4. Key quotations

FINAL ANSWER:
"""

print("\n================ AI ANSWER ================\n")

with tempfile.NamedTemporaryFile(
    delete=False,
    mode="w",
    suffix=".txt"
) as f:

    f.write(prompt)

    temp_prompt = f.name

command = [
    str(LLAMA_BIN),
    "-m",
    str(MODEL_PATH),
    "-f",
    temp_prompt,
    "-n",
    "600",
    "-t",
    "4",
    "-c",
    "4096"
]

result = subprocess.run(
    command,
    capture_output=True,
    text=True
)

print(result.stdout)

print("\n================ SOURCES ================\n")

for s in sources:
    print("-", s)

