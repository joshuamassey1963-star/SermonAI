from pathlib import Path
import sqlite3

DB_PATH = Path.home() / "SermonAI/data/sermons.db"
CHUNK_DB = Path.home() / "SermonAI/data/chunks.db"

CHUNK_SIZE = 500
OVERLAP = 100

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

chunk_conn = sqlite3.connect(CHUNK_DB)
chunk_cursor = chunk_conn.cursor()

chunk_cursor.execute("""
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sermon_title TEXT,
    sermon_code TEXT,
    chunk_index INTEGER,
    chunk_text TEXT
)
""")

cursor.execute("""
SELECT title, code, text
FROM sermons
""")

sermons = cursor.fetchall()

print(f"Found sermons: {len(sermons)}")

total_chunks = 0

for sermon_index, sermon in enumerate(sermons, start=1):

    title, code, text = sermon

    print(f"\n[{sermon_index}/{len(sermons)}] {title}")

    words = text.split()

    start = 0
    chunk_index = 0

    while start < len(words):

        end = start + CHUNK_SIZE

        chunk_words = words[start:end]

        chunk_text = " ".join(chunk_words)

        chunk_cursor.execute("""
        INSERT INTO chunks (
            sermon_title,
            sermon_code,
            chunk_index,
            chunk_text
        )
        VALUES (?, ?, ?, ?)
        """, (
            title,
            code,
            chunk_index,
            chunk_text
        ))

        total_chunks += 1

        chunk_index += 1

        start += CHUNK_SIZE - OVERLAP

chunk_conn.commit()

conn.close()
chunk_conn.close()

print(f"\nTotal chunks created: {total_chunks}")
print(f"Chunk database saved: {CHUNK_DB}")
