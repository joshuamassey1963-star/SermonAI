import sqlite3
import numpy as np
from pathlib import Path
from collections import defaultdict
from fastembed import TextEmbedding

CHUNK_DB = Path.home() / "SermonAI/data/chunks.db"

VECTOR_PATH = Path.home() / "SermonAI/data/embeddings/vectors.npy"
CHUNK_IDS_PATH = Path.home() / "SermonAI/data/embeddings/chunk_ids.npy"

TOP_K = 10

print("Loading embedding model...")

embedding_model = TextEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)

print("Loading vectors...")

vectors = np.load(VECTOR_PATH)
chunk_ids = np.load(CHUNK_IDS_PATH)

conn = sqlite3.connect(CHUNK_DB)
cursor = conn.cursor()

query = input("\nHybrid Search Query: ").strip()

print("\nGenerating semantic embedding...")

query_vector = list(embedding_model.embed([query]))[0]
query_vector = np.array(query_vector, dtype=np.float32)

print("Calculating semantic similarities...")

dot_products = np.dot(vectors, query_vector)

vector_norms = np.linalg.norm(vectors, axis=1)
query_norm = np.linalg.norm(query_vector)

semantic_scores = dot_products / (vector_norms * query_norm)

semantic_top_indices = np.argsort(semantic_scores)[-50:]

combined_scores = defaultdict(float)

for idx in semantic_top_indices:

    chunk_id = int(chunk_ids[idx])

    score = float(semantic_scores[idx])

    combined_scores[chunk_id] += score * 0.7

print("Running keyword search...")

search_term = f"%{query}%"

cursor.execute("""
SELECT
    id,
    sermon_title,
    sermon_code,
    chunk_text
FROM chunks
WHERE chunk_text LIKE ?
LIMIT 50
""", (search_term,))

keyword_results = cursor.fetchall()

for row in keyword_results:

    chunk_id = row[0]

    combined_scores[chunk_id] += 0.3

print("\n=== HYBRID SEARCH RESULTS ===\n")

ranked_results = sorted(
    combined_scores.items(),
    key=lambda x: x[1],
    reverse=True
)

shown = 0

for chunk_id, final_score in ranked_results:

    cursor.execute("""
    SELECT
        sermon_title,
        sermon_code,
        chunk_text
    FROM chunks
    WHERE id = ?
    """, (chunk_id,))

    row = cursor.fetchone()

    if not row:
        continue

    title, code, chunk_text = row

    shown += 1

    print("=" * 80)
    print(f"Result #{shown}")
    print(f"Combined Score : {final_score:.4f}")
    print(f"Sermon         : {title}")
    print(f"Code            : {code}")

    print("\nChunk:\n")

    print(chunk_text[:1500])

    print("\n")

    if shown >= TOP_K:
        break

conn.close()
