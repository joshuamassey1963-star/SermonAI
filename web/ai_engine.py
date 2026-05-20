import sqlite3
import numpy as np
from fastembed import TextEmbedding
from llama_cpp import Llama

DB_PATH = "/data/data/com.termux/files/home/SermonAI/data/chunks.db"
VECTORS_PATH = "/data/data/com.termux/files/home/SermonAI/data/embeddings/vectors.npy"
CHUNK_IDS_PATH = "/data/data/com.termux/files/home/SermonAI/data/embeddings/chunk_ids.npy"
MODEL_PATH = "/data/data/com.termux/files/home/SermonAI/models/gemma-3-4b-it-Q4_K_M.gguf"

print("\n[AI] Loading embedding model...\n")
embedding_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

print("[AI] Loading vectors...\n")
vectors = np.load(VECTORS_PATH)
chunk_ids = np.load(CHUNK_IDS_PATH)
print(f"[AI] Loaded {len(vectors)} vectors")

conn = sqlite3.connect(DB_PATH, check_same_thread=False)
cursor = conn.cursor()

print("\n[AI] Loading Gemma...\n")
llm = Llama(model_path=MODEL_PATH, n_ctx=4096, n_threads=4, n_gpu_layers=0, verbose=False)

# SENIOR DEV FIX: Safely expanded limits. 45,000 chars is perfect for deep doctrinal pulls without API timeouts.
MODES = {
    "short": {"top_k": 15, "chunk_size": 800, "tokens": 1000, "max_chars": 15000},
    "medium": {"top_k": 30, "chunk_size": 1000, "tokens": 2500, "max_chars": 30000},
    "long": {"top_k": 45, "chunk_size": 1200, "tokens": 4000, "max_chars": 45000} 
}

def cosine_similarity(query_vector, vectors):
    query_norm = np.linalg.norm(query_vector)
    vector_norms = np.linalg.norm(vectors, axis=1)
    return np.dot(vectors, query_vector) / (vector_norms * query_norm + 1e-10)

def retrieve_sermon_evidence(query, mode="medium"):
    if mode not in MODES: mode = "medium"
    config = MODES[mode]
    
    query_embedding = list(embedding_model.embed([query]))[0]
    scores = cosine_similarity(query_embedding, vectors)
    
    POOL_SIZE = 1000 
    top_indices = np.argsort(scores)[-POOL_SIZE:][::-1]

    ignore_words = {"what", "when", "where", "how", "brother", "branham", "explain", "hindi", "give", "detail", "about", "should", "could", "would", "that", "this", "can", "did", "does"}
    query_terms = set([w.strip("?.,!") for w in query.lower().split() if len(w) > 3 and w not in ignore_words])

    raw_results = []
    for idx in top_indices:
        chunk_id = int(chunk_ids[idx])
        cursor.execute("SELECT sermon_title, sermon_code, chunk_text FROM chunks WHERE id = ?", (chunk_id,))
        row = cursor.fetchone()
        if not row: continue
        
        title = row[0] if row[0] else "Unknown Title"
        code = row[1] if row[1] else "00-0000"
        text = row[2] if row[2] else ""
        base_score = scores[idx]
        
        year = 1947
        try:
            yy_str = code[:2]
            if yy_str.isdigit():
                yy = int(yy_str)
                if 47 <= yy <= 99: year = 1900 + yy
                elif 0 <= yy <= 10: year = 2000 + yy
        except: pass
        
        year_bonus = (year - 1947) * 0.008
        
        text_lower = text.lower()
        lexical_bonus = 0.0
        for term in query_terms:
            if term in text_lower:
                lexical_bonus += 0.04 
                
        final_score = base_score * (1.0 + year_bonus + lexical_bonus)
        
        raw_results.append({
            "title": title,
            "code": code,
            "text": (text.replace("\n", " ").strip())[:config["chunk_size"]],
            "final_score": final_score
        })

    raw_results.sort(key=lambda x: x["final_score"], reverse=True)
    final_pool = raw_results[:config["top_k"]]
    final_pool.sort(key=lambda x: str(x["code"]))

    evidence_blocks = []
    sources = []
    for r in final_pool:
        sources.append({"title": r["title"], "code": r["code"]})
        evidence_blocks.append(f"SOURCE: {r['title']}\nCODE: {r['code']}\nQUOTE: {r['text']}")

    evidence_text = "\n\n---\n\n".join(evidence_blocks)
    return {"evidence": evidence_text[:config["max_chars"]], "sources": sources, "config": config}

def offline_rag_search(query, mode="medium"):
    retrieval = retrieve_sermon_evidence(query, mode)
    prompt = f"Answer strictly based ONLY on evidence. Give detailed response if long mode.\n\nQUESTION: {query}\n\nEVIDENCE:\n{retrieval['evidence'][:9000]}\n\nANSWER:"
    output = llm(prompt, max_tokens=retrieval["config"]["tokens"], temperature=0.2, stop=["</s>"])
    return {"answer": output["choices"][0]["text"], "sources": retrieval["sources"]}
