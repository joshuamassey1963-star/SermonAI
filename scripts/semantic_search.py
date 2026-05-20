import sqlite3
import numpy as np
from pathlib import Path
from fastembed import TextEmbedding

DB_PATH = Path.home() / "SermonAI/data/chunks.db"

VECTOR_PATH = Path.home() / "SermonAI/data/embeddings/vectors.npy"
CHUNK_IDS_PATH = Path.home() / "SermonAI/data/embeddings/chunk_ids.npy"

TOP_K = 5

print("Loading embedding model...")

embedding_model = TextEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)

print("Loading vectors...")

vectors = np.load(VECTOR_PATH)
chunk_ids = np.load(CHUNK_IDS_PATH)

print(f"Loaded vectors: {vectors.shape}")

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

query = input("\nSemantic Search Query: ").strip()

print("\nGenerating query embedding...")

query_vector = list(embedding_model.embed([query]))[0]

query_vector = np.array(query_vector, dtype=np.float32)

print("Calculating similarities...")

dot_products = np.dot(vectors, query_vector)

vector_norms = np.linalg.norm(vectors, axis=1)
query_norm = np.linalg.norm(query_vector)

similarities = dot_products / (vector_norms * query_norm)

top_indices = np.argsort(similarities)[-TOP_K:][::-1]

print("\n=== TOP SEMANTIC RESULTS ===\n")

for rank, idx in enumerate(top_indices, start=1):

    chunk_id = int(chunk_ids[idx])

    similarity = similarities[idx]

    cursor.execute("""
    SELECT
        sermon_title,
        sermon_code,
        chunk_text
    FROM chunks
    WHERE id = ?
    """, (chunk_id,))

    row = cursor.fetchone()

    if row:

        title, code, chunk_text = row

        print("=" * 80)
        print(f"Result #{rank}")
        print(f"Similarity : {similarity:.4f}")
        print(f"Sermon    : {title}")
        print(f"Code       : {code}")

        print("\nChunk:\n")

        print(chunk_text[:1500])

        print("\n")

conn.close()
