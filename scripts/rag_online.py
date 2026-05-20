import os
import re
import json
import sqlite3
import requests
import numpy as np
from fastembed import TextEmbedding

# =========================================
# CONFIG
# =========================================

DB_PATH = os.path.expanduser(
    "~/SermonAI/data/chunks.db"
)

VECTORS_PATH = os.path.expanduser(
    "~/SermonAI/data/embeddings/vectors.npy"
)

CHUNK_IDS_PATH = os.path.expanduser(
    "~/SermonAI/data/embeddings/chunk_ids.npy"
)

API_KEY = os.getenv(
    "OPENROUTER_API_KEY"
)

MODEL_NAME = (
    "deepseek/deepseek-chat-v3-0324"
)

TOP_K = 12

# =========================================
# CHECK API KEY
# =========================================

if not API_KEY:
    raise ValueError(
        "OPENROUTER_API_KEY not found"
    )

# =========================================
# COSINE SIMILARITY
# =========================================

def cosine_similarity_manual(
    query_vector,
    vectors
):

    query_norm = np.linalg.norm(
        query_vector
    )

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

# =========================================
# ANSWER MODE
# =========================================

def detect_mode(query):

    q = query.lower()

    short_words = [
        "summary",
        "short",
        "brief"
    ]

    sermon_words = [
        "deep",
        "details",
        "explain",
        "clarification"
    ]

    for word in short_words:

        if word in q:
            return "short"

    for word in sermon_words:

        if word in q:
            return "sermon"

    return "long"

# =========================================
# LOAD EMBEDDING MODEL
# =========================================

print(
    "\nLoading embedding model...\n"
)

embedding_model = TextEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)

# =========================================
# LOAD VECTORS
# =========================================

print(
    "Loading vectors...\n"
)

vectors = np.load(
    VECTORS_PATH
)

chunk_ids = np.load(
    CHUNK_IDS_PATH
)

print(
    f"Loaded {len(vectors)} vectors"
)

# =========================================
# SQLITE
# =========================================

conn = sqlite3.connect(
    DB_PATH
)

cursor = conn.cursor()

# =========================================
# MEMORY
# =========================================

conversation_history = []

last_answer = ""

# =========================================
# CHAT LOOP
# =========================================

while True:

    query = input(
        "\nYou:\n> "
    ).strip()

    # =====================================
    # EXIT
    # =====================================

    if query.lower() in [
        "exit",
        "quit"
    ]:

        print(
            "\nGoodbye.\n"
        )

        break

    # =====================================
    # CONTINUE
    # =====================================

    if query.lower() == "continue":

        query = (
            "Continue previous answer "
            "with more details."
        )

    mode = detect_mode(query)

    print(
        f"\nAnswer Mode: {mode}\n"
    )

    # =====================================
    # AI QUERY EXPANSION
    # =====================================

    print(
        "Understanding doctrine...\n"
    )

    expand_prompt = f"""
Expand this doctrine question into
related sermon concepts.

Return ONLY expanded search concepts.

Question:
{query}
"""

    expand_headers = {
        "Authorization":
            f"Bearer {API_KEY}",
        "Content-Type":
            "application/json"
    }

    expand_data = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": expand_prompt
            }
        ],
        "temperature": 0.1,
        "max_tokens": 80
    }

    expanded_query = query

    try:

        expand_response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=expand_headers,
            data=json.dumps(expand_data),
            timeout=120
        )

        expanded_query = (
            expand_response.json()
            ["choices"][0]
            ["message"]["content"]
            .strip()
        )

    except:
        pass

    print(
        "Expanded Query:\n"
    )

    print(
        expanded_query
    )

    print()

    # =====================================
    # EMBEDDING
    # =====================================

    print(
        "Generating embedding...\n"
    )

    query_embedding = list(
        embedding_model.embed(
            [expanded_query]
        )
    )[0]

    # =====================================
    # SEARCH
    # =====================================

    print(
        "Searching sermons...\n"
    )

    scores = cosine_similarity_manual(
        query_embedding,
        vectors
    )

    top_indices = np.argsort(
        scores
    )[-TOP_K:][::-1]

    # =====================================
    # BUILD EVIDENCE
    # =====================================

    evidence_blocks = []

    retrieved_sources = []

    seen = set()

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

        (
            sermon_title,
            sermon_code,
            chunk_text
        ) = row

        cleaned = (
            chunk_text
            .replace("\n", " ")
            .strip()
        )

        cleaned = re.sub(
            r"\s+",
            " ",
            cleaned
        )

        cleaned = cleaned[:600]

        if cleaned in seen:
            continue

        seen.add(cleaned)

        evidence = f"""
SOURCE:
{sermon_title}

SERMON:
{sermon_code}

QUOTE:
{cleaned}
"""

        evidence_blocks.append(
            evidence
        )

        retrieved_sources.append({
            "title": sermon_title,
            "code": sermon_code,
            "score": scores[idx]
        })

    full_evidence = "\n\n".join(
        evidence_blocks
    )

    # =====================================
    # MODE PROMPTS
    # =====================================

    mode_prompt = ""

    if mode == "short":

        mode_prompt = """
Give short simple answer.

Keep under 120 words.
"""

    elif mode == "sermon":

        mode_prompt = """
Explain deeply like a sermon.

Give long detailed teaching.
"""

    else:

        mode_prompt = """
Give detailed natural explanation.
"""

    # =====================================
    # MEMORY CONTEXT
    # =====================================

    memory_text = "\n".join(
        conversation_history[-6:]
    )

    # =====================================
    # FINAL PROMPT
    # =====================================

    prompt = f"""
Answer ONLY from sermon evidence.

Do NOT use outside theology.

Do NOT invent doctrine.

Use natural conversational style.

Continue conversation naturally.

PREVIOUS CONVERSATION:
{memory_text}

{mode_prompt}

QUESTION:
{query}

SERMON EVIDENCE:
{full_evidence}
"""

    # =====================================
    # FINAL AI CALL
    # =====================================

    print(
        "Generating doctrinal answer...\n"
    )

    final_headers = {
        "Authorization":
            f"Bearer {API_KEY}",
        "Content-Type":
            "application/json"
    }

    final_data = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.1,
        "max_tokens": 1600
    }

    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers=final_headers,
        data=json.dumps(final_data),
        timeout=180
    )

    if response.status_code != 200:

        print("\nERROR:\n")

        print(response.text)

        continue

    result = response.json()

    answer = result["choices"][0][
        "message"
    ]["content"]

    last_answer = answer

    # =====================================
    # SAVE MEMORY
    # =====================================

    conversation_history.append(
        f"User: {query}"
    )

    conversation_history.append(
        f"Assistant: {answer[:1200]}"
    )

    # =====================================
    # OUTPUT
    # =====================================

    print(
        "\n====================================\n"
    )

    print(answer)

    print(
        "\n===================================="
    )

    print(
        "SOURCES"
    )

    print(
        "====================================\n"
    )

    for i, source in enumerate(
        retrieved_sources,
        start=1
    ):

        print(
            f"{i}. "
            f"{source['title']}"
        )

        print(
            f"Code: "
            f"{source['code']}"
        )

        print(
            f"Similarity: "
            f"{source['score']:.4f}\n"
        )

conn.close()
