import os
import requests
from dotenv import load_dotenv

from web.ai_engine import retrieve_sermon_evidence

load_dotenv("/data/data/com.termux/files/home/SermonAI/.env", override=True)
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL_NAME = "deepseek/deepseek-chat"

online_chat_history = []

def ask_online_ai(prompt, max_tokens=700, temperature=0.1):
    try:
        response = requests.post(
            OPENROUTER_URL,
            headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"},
            json={"model": MODEL_NAME, "messages": [{"role": "user", "content": prompt}], "temperature": temperature, "max_tokens": max_tokens},
            timeout=200
        )
        data = response.json()
        if "choices" not in data: return {"success": False, "answer": "API Error: No response or connection timeout."}
        
        content = data["choices"][0]["message"]["content"]
        if not content or content.strip() == "": return {"success": False, "answer": "Error: Empty response from AI."}
        return {"success": True, "answer": content}
    except Exception as e:
        return {"success": False, "answer": f"Request Failed/Timeout: {str(e)}"}

def generate_search_query(query, history):
    if not history:
        prompt = f"Extract core theological search keywords from this query. Output ONLY English keywords.\nQuery: {query}"
    else:
        hist_str = "\n".join([f"User: {c['user']}" for c in history[-2:]])
        prompt = f"Context:\n{hist_str}\n\nFollow-up: {query}\n\nRewrite follow-up into standalone English keywords combining context. Output ONLY keywords."
    
    res = ask_online_ai(prompt, max_tokens=50, temperature=0.0)
    if res["success"]: return res["answer"].replace('"', '').strip()
    return query

def online_search(query, mode="medium"):
    global online_chat_history
    
    english_query = generate_search_query(query, online_chat_history)
    
    # SENIOR DEV FIX: PREVENTING KEYWORD LOSS
    # Combining the raw user query with AI keywords so exact words (like "eggs") are NEVER skipped.
    combined_search_vector = f"{query} {english_query}"
    print(f"\n[AI] Search Vector: {combined_search_vector}\n")

    retrieval = retrieve_sermon_evidence(query=combined_search_vector, mode=mode)
    evidence = retrieval["evidence"]
    max_tokens = retrieval["config"]["tokens"]

    history_context = ""
    if online_chat_history:
        history_context = "PAST CONVERSATION:\n"
        for chat in online_chat_history: 
            history_context += f"User: {chat['user']}\nAI: {chat['ai'][:200]}...\n\n"

    # Dynamic length instruction
    if mode == "long":
        length_instruction = "Provide a HIGHLY DETAILED and EXHAUSTIVE response. Quote extensively and break down every aspect of the topic."
    elif mode == "short":
        length_instruction = "Provide a brief, concise summary."
    else:
        length_instruction = "Provide a balanced and moderately detailed explanation."

    prompt = f"""You are an advanced theological research assistant for William Branham's teachings.

CRITICAL DOCTRINAL REASONING RULES:
1. THE TIMELINE: The evidence below is strictly ordered chronologically (oldest to newest).
2. DOCTRINAL EVOLUTION: Brother Branham's teachings progressed and refined over time. Explain this progression clearly.
3. WARNINGS vs. SOLUTIONS (CRITICAL): If he mentions a prophecy, warning, or danger (e.g., diseases, poisoned food, not eating eggs, valleys) but ALSO provides a Biblical solution or action for believers (e.g., sanctifying it by prayer, faith, and thanksgiving), YOU MUST EXPLAIN BOTH. 
4. MATURE POSITION: Treat his chronologically latest explicit statement on the specific topic as his mature, final doctrinal position.
5. FORMATTING: Use `## ` or `### ` for Headings. Use `>` for Quotes. Use `**` for Highlights. 
6. RESPONSE LENGTH: {length_instruction}
7. NO HALLUCINATION. Base everything strictly on the evidence below.

{history_context}

CURRENT QUESTION:
{query}

SERMON EVIDENCE:
{evidence}

ANSWER:"""

    result = ask_online_ai(prompt, max_tokens)
    
    final_answer_text = result["answer"]
    if not result["success"]: 
        final_answer_text = f"⚠️ **Connection/API Error:**\n{result['answer']}\n\n*The server took too long or failed to respond. Your chat history is safe. Please click 'Follow Up' to try asking again.*"

    online_chat_history.append({"user": query, "ai": final_answer_text})
    if len(online_chat_history) > 5: online_chat_history.pop(0)

    return {"answer": final_answer_text, "sources": retrieval["sources"]}

def hybrid_search(query, mode="medium"):
    return {"answer": "Hybrid temporarily disabled.", "sources": []}
