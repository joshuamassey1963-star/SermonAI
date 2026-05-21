import sqlite3
import numpy as np
from fastembed import TextEmbedding
from llama_cpp import Llama

DB_PATH = "/data/data/com.termux/files/home/SermonAI/data/chunks.db"
VECTORS_PATH = "/data/data/com.termux/files/home/SermonAI/data/embeddings/vectors.npy"
CHUNK_IDS_PATH = "/data/data/com.termux/files/home/SermonAI/data/embeddings/chunk_ids.npy"
MODEL_PATH = "/data/data/com.termux/files/home/SermonAI/models/gemma-3-4b-it-Q4_K_M.gguf"

embedding_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
vectors = np.load(VECTORS_PATH)
chunk_ids = np.load(CHUNK_IDS_PATH)
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
cursor = conn.cursor()

# n_threads=4 threads ka istemal kar raha hai, n_ctx=2048 rakha hai
llm = Llama(model_path=MODEL_PATH, n_ctx=2048, n_threads=4, verbose=False)

# Speed Tweaks: top_k aur tokens kam kiye taaki 60 seconds ke andar result mile
MODES = {
    "short": {"top_k": 3, "chunk_size": 300, "tokens": 250, "max_chars": 1000},
    "medium": {"top_k": 4, "chunk_size": 400, "tokens": 400, "max_chars": 2000},
    "long": {"top_k": 6, "chunk_size": 500, "tokens": 600, "max_chars": 3500}
}

def cosine_similarity(query_vector, vectors):
    return np.dot(vectors, query_vector) / (np.linalg.norm(vectors, axis=1) * np.linalg.norm(query_vector) + 1e-10)

def retrieve_sermon_evidence(query, mode="medium"):
    config = MODES.get(mode, MODES["medium"])
    query_embedding = list(embedding_model.embed([query]))[0]
    scores = cosine_similarity(query_embedding, vectors)
    top_indices = np.argsort(scores)[-200:][::-1] # 200 chunk search is enough
    
    raw_results = []
    for idx in top_indices:
        cursor.execute("SELECT sermon_title, sermon_code, chunk_text FROM chunks WHERE id = ?", (int(chunk_ids[idx]),))
        row = cursor.fetchone()
        if row:
            raw_results.append({"title": row[0], "code": row[1], "text": row[2][:config["chunk_size"]]})
        if len(raw_results) >= config["top_k"]: break
    
    evidence = "\n\n".join([f"SOURCE: {r['title']} ({r['code']})\n{r['text']}" for r in raw_results])
    return {"evidence": evidence, "sources": [{"title": r['title'], "code": r['code']} for r in raw_results], "config": config}

def offline_rag_search(query, mode="medium"):
    try:
        retrieval = retrieve_sermon_evidence(query, mode)
        # prompt brevity: jitna chota utna fast
        prompt = f"Context: {retrieval['evidence']}\n\nQ: {query}\n\nA:"
        output = llm(prompt, max_tokens=retrieval["config"]["tokens"], temperature=0.3, stop=["</s>"])
        return {"answer": output["choices"][0]["text"], "sources": retrieval["sources"]}
    except Exception as e:
        return {"answer": f"Error: {str(e)}", "sources": []}
