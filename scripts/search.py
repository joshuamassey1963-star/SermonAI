import sqlite3
from pathlib import Path

DB_PATH = Path.home() / "SermonAI/data/sermons.db"

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

print("\n=== SERMON SEARCH ENGINE ===\n")

query = input("Search: ").strip()

sql = """
SELECT
    title,
    code,
    filename,
    substr(text, 1, 500)
FROM sermons
WHERE
    title LIKE ?
    OR text LIKE ?
LIMIT 10
"""

search_term = f"%{query}%"

cursor.execute(sql, (search_term, search_term))

results = cursor.fetchall()

print(f"\nResults found: {len(results)}\n")

for index, row in enumerate(results, start=1):

    title, code, filename, snippet = row

    print("=" * 80)
    print(f"Result #{index}")
    print(f"Title: {title}")
    print(f"Code : {code}")
    print(f"File : {filename}")
    print("\nSnippet:\n")
    print(snippet[:500])
    print("\n")

conn.close()
