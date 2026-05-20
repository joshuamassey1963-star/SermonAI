from pathlib import Path
import sqlite3
import re

TEXT_DIR = Path.home() / "SermonAI/data/output"
DB_PATH = Path.home() / "SermonAI/data/sermons.db"

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS sermons (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    code TEXT,
    year INTEGER,
    title TEXT,
    filename TEXT,
    text TEXT
)
""")

def parse_filename(filename):
    stem = Path(filename).stem

    match = re.match(r'(\d{2})_(\d{4}[A-Z]?)_(.+)', stem)

    if match:
        year_short = match.group(1)
        code_part = match.group(2)
        title_part = match.group(3)

        year = int("19" + year_short)

        code = f"{year_short}_{code_part}"

        title = title_part.replace("_", " ")

        return code, year, title

    return None, None, stem

txt_files = sorted(TEXT_DIR.glob("*.txt"))

print(f"Found TXT files: {len(txt_files)}")

for index, txt_file in enumerate(txt_files, start=1):

    print(f"[{index}/{len(txt_files)}] {txt_file.name}")

    text = txt_file.read_text(encoding="utf-8", errors="ignore")

    code, year, title = parse_filename(txt_file.name)

    cursor.execute("""
    INSERT INTO sermons (
        code,
        year,
        title,
        filename,
        text
    )
    VALUES (?, ?, ?, ?, ?)
    """, (
        code,
        year,
        title,
        txt_file.name,
        text
    ))

conn.commit()
conn.close()

print("\nDatabase build complete.")
print(f"Saved database: {DB_PATH}")
