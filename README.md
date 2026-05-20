# SermonAI: Intelligent Sermon Retrieval System

SermonAI ek AI-powered system hai jo RAG (Retrieval-Augmented Generation) technology ka use karta hai. Yeh project aapke sermons ko index karta hai aur local Gemma model ke dwara unhe samajh kar jawab deta hai.

---

## 🛠 Project Structure (Memory Tree)
Aapke folder mein sab kuch is tarah organized hai:
- **/web**: Ismein aapka backend () aur AI engine () hai.
- **/data**: Ismein aapka sermons ka database () aur vector embeddings hain.
- **/models**: Ismein aapka local LLM model () rakha jata hai.
- **/embed_env**: Ismein aapka project ka support system (Python virtual environment) hai.

---

## 🚀 Step-by-Step Setup Guide

### 1. Repository Clone Karein
Sabse pehle code ko download karein:
```bash
git clone https://github.com/joshuamassey1963-star/SermonAI.git
cd SermonAI
```

### 2. Large Data & Models Download Karein
Code ke sath data folder nahi aata, isliye use Google Drive se download karna hoga:
```bash
pip install gdown
gdown --id 11upUdnZSQ8E9nfMbyWN_A3pxN8Tym_SN --output backup.tar.gz
tar -xzvf backup.tar.gz
```
*(Yeh command 'data' aur 'models' folder ko automatically sahi jagah extract kar degi)*

### 3. Environment Setup (The Foundation)
Project chalane ke liye zaroori libraries setup karein:
```bash
# Backup se environment restore karein (agar available ho)
tar -xzvf sermon_env_backup.tar.gz

# Environment activate karein
source embed_env/venv/bin/activate
```

### 4. Server Start Karein
Ab aapka system ready hai. Server chalaane ke liye ye command dein:
```bash
uvicorn web.app:app --host 0.0.0.0 --port 8000
```
Ab aap apne browser mein `http://0.0.0.0:8000` par jakar sermon search kar sakte hain.

---

## 💾 Maintenance & Backup
- **Code:** Git ke through sync rakhein (`git push`).
- **Memory (Data/Models):** Inka backup periodic Google Drive par upload karte rahein.
- **Environment:** Agar naya environment banayein, toh `pip freeze > requirements.txt` jarur karein.
