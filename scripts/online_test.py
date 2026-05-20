import os
import requests
import json

API_KEY = os.getenv("OPENROUTER_API_KEY")

if not API_KEY:
    raise ValueError("OPENROUTER_API_KEY not found")

url = "https://openrouter.ai/api/v1/chat/completions"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

question = "What is serpent seed?"

evidence = """
DOCTRINE: Serpent Seed

EVIDENCE 1
SOURCE: The Serpent's Seed (1958)

QUOTE:
"The serpent seed brought forth Cain."

POINT:
Cain originated through the serpent lineage doctrine.

---

EVIDENCE 2
SOURCE: Questions and Answers

QUOTE:
"I will put enmity between thy seed and her seed."

POINT:
Two seed lines are contrasted in Eden.
"""

prompt = f"""
You are a theological research assistant.

Use ONLY the provided evidence.

Do NOT invent information.

Answer doctrinally and clearly.

QUESTION:
{question}

EVIDENCE:
{evidence}
"""

data = {
    "model": "deepseek/deepseek-chat-v3-0324",
    "messages": [
        {
            "role": "user",
            "content": prompt
        }
    ],
    "temperature": 0.3,
    "max_tokens": 700
}

print("\\nSending request to DeepSeek V3...\\n")

response = requests.post(
    url,
    headers=headers,
    data=json.dumps(data),
    timeout=120
)

print("STATUS:", response.status_code)

if response.status_code != 200:
    print("\\nERROR RESPONSE:\\n")
    print(response.text)
    exit()

result = response.json()

answer = result["choices"][0]["message"]["content"]

print("\\n===== AI ANSWER =====\\n")
print(answer)
