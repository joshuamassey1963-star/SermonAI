import os
import requests
from dotenv import load_dotenv
from web.ai_engine import retrieve_sermon_evidence

load_dotenv("/data/data/com.termux/files/home/SermonAI/.env", override=True)
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL_NAME = "deepseek/deepseek-chat"

online_chat_history = []

def ask_online_ai(prompt, max_tokens=1500, temperature=0.1):
    try:
        response = requests.post(
            OPENROUTER_URL,
            headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"},
            json={"model": MODEL_NAME, "messages": [{"role": "user", "content": prompt}], "temperature": temperature, "max_tokens": max_tokens},
            timeout=200
        )
        data = response.json()
        if "choices" not in data: return {"success": False, "answer": "API Error."}
        return {"success": True, "answer": data["choices"][0]["message"]["content"]}
    except Exception as e:
        return {"success": False, "answer": str(e)}

def online_search(query, mode="medium"):
    global online_chat_history
    retrieval = retrieve_sermon_evidence(query=query, mode=mode)
    
    prompt = f"""You are an advanced researcher of William Branham's sermons.
    
    CRITICAL INSTRUCTIONS:
    1. EXACT MATCHES: If the user asks for specific quotes, words, or sentences, you MUST provide them verbatim from the EVIDENCE below.
    2. TIMELINE TRACKING: Synthesize teachings chronologically. If a doctrine evolved, explicitly highlight the transition and the final mature position.
    3. NO HALLUCINATION: Everything must be grounded in the provided Evidence blocks.
    4. STRUCTURE: Use headings for sections, > for quotes, and bold for key theological concepts.

    EVIDENCE:
    {retrieval['evidence']}
    
    USER QUERY: {query}
    
    RESEARCH RESPONSE:"""

    result = ask_online_ai(prompt, max_tokens=retrieval["config"]["tokens"] * 2)
    answer = result["answer"] if result["success"] else "Error communicating with Online Engine."
    
    online_chat_history.append({"user": query, "ai": answer})
    if len(online_chat_history) > 5: online_chat_history.pop(0)
    
    return {"answer": answer, "sources": retrieval["sources"]}

def hybrid_search(query, mode="medium"):
    # Future expansion ke liye placeholder
    return {"answer": "Hybrid search is currently under maintenance.", "sources": []}
