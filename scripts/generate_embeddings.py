import sqlite3
import numpy as np
from pathlib import Path
from fastembed import TextEmbedding

DB_PATH = Path.home() / "SermonAI/data/chunks.db"

EMBED_DIR = Path.home() / "SermonAI/data/embeddings"
EMBED_DIR.mkdir(parents=True, exist_ok=True)

VECTORS_PATH = EMBED_DIR / "vectors.npy"
CHUNK_IDS_PATH = EMBED_DIR / "chunk_ids.npy"

BATCH_SIZE = 8

print("Loading embedding model...")

embedding_model = TextEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)

print("Connecting to chunk database...")

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

cursor.execute("""
SELECT id, chunk_text
FROM chunks
""")

rows = cursor.fetchall()

print(f"Total chunks loaded: {len(rows)}")

chunk_ids = []
all_vectors = []

for start in range(0, len(rows), BATCH_SIZE):

    batch = rows[start:start + BATCH_SIZE]

    texts = [row[1] for row in batch]
    ids = [row[0] for row in batch]

    print(f"Processing batch {start} -> {start + len(batch)}")

    embeddings = list(embedding_model.embed(texts))

    for chunk_id, vector in zip(ids, embeddings):

        chunk_ids.append(chunk_id)
        all_vectors.append(vector)

vectors_array = np.array(all_vectors, dtype=np.float32)
chunk_ids_array = np.array(chunk_ids)

np.save(VECTORS_PATH, vectors_array)
np.save(CHUNK_IDS_PATH, chunk_ids_array)

print("\nEmbedding generation complete.")
print(f"Vectors shape: {vectors_array.shape}")
print(f"Saved vectors: {VECTORS_PATH}")
print(f"Saved chunk IDs: {CHUNK_IDS_PATH}")
