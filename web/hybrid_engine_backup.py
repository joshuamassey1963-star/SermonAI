import os
import requests
from dotenv import load_dotenv

from web.ai_engine import offline_rag_search

# =========================
# ENV
# =========================

load_dotenv(
    "/data/data/com.termux/files/home/SermonAI/.env"
)

OPENROUTER_API_KEY = os.getenv(
    "OPENROUTER_API_KEY"
)

OPENROUTER_URL = (
    "https://openrouter.ai/api/v1/chat/completions"
)

MODEL_NAME = "openai/gpt-3.5-turbo"

# =========================
# SAFE REQUEST
# =========================

def ask_online_ai(prompt):

    try:

        response = requests.post(

            OPENROUTER_URL,

            headers={

                "Authorization":
                f"Bearer {OPENROUTER_API_KEY}",

                "Content-Type":
                "application/json",

                "HTTP-Referer":
                "http://localhost:8000",

                "X-Title":
                "OfflineSermonAI"

            },

            json={

                "model": MODEL_NAME,

                "messages": [

                    {
                        "role": "user",
                        "content": prompt
                    }

                ],

                "temperature": 0.2,
                "max_tokens": 700

            },

            timeout=120

        )

        print("\n[OPENROUTER STATUS]")
        print(response.status_code)

        data = response.json()

        print("\n[OPENROUTER RESPONSE]")
        print(data)

        # SAFE CHECK

        if "choices" not in data:

            error_message = (
                data.get("error", {})
                .get("message", "Unknown API Error")
            )

            return {
                "success": False,
                "answer": error_message
            }

        answer = (
            data["choices"][0]["message"]["content"]
        )

        return {
            "success": True,
            "answer": answer
        }

    except Exception as e:

        return {
            "success": False,
            "answer": str(e)
        }

# =========================
# ONLINE SEARCH
# =========================

def online_search(
    query,
    mode="medium"
):

    result = ask_online_ai(query)

    if not result["success"]:

        return {

            "answer":
            f"Online AI Error:\n\n{result['answer']}",

            "sources": []

        }

    return {

        "answer":
        result["answer"],

        "sources": [

            {
                "title": "DeepSeek Online",
                "code": "ONLINE AI"
            }

        ]

    }

# =========================
# HYBRID SEARCH
# =========================

def hybrid_search(
    query,
    mode="medium"
):

    improve_prompt = f"""
Improve this doctrinal search query
for semantic sermon retrieval.

Keep original meaning.

Query:
{query}
"""

    online_result = ask_online_ai(
        improve_prompt
    )

    # IF ONLINE FAILS
    # FALLBACK TO OFFLINE

    if not online_result["success"]:

        result = offline_rag_search(
            query,
            mode
        )

        result["answer"] = (

            "[Hybrid Fallback Mode]\n\n"

            "Online AI unavailable.\n\n"

            + result["answer"]
        )

        return result

    improved_query = (
        online_result["answer"]
        .strip()
    )

    result = offline_rag_search(
        improved_query,
        mode
    )

    result["sources"].append({

        "title": "Hybrid AI",
        "code": "DeepSeek + Gemma"

    })

    return result
