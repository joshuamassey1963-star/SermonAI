import sqlite3
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from web.ai_engine import offline_rag_search
from web.hybrid_engine import online_search, hybrid_search, online_chat_history

app = FastAPI()
app.mount("/static", StaticFiles(directory="web/static"), name="static")
templates = Jinja2Templates(directory="web/templates")

SERMONS_DB_PATH = "/data/data/com.termux/files/home/SermonAI/data/sermons.db"

def get_db_connection():
    conn = sqlite3.connect(SERMONS_DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

# --- PHASE 1: CHAT ROUTES (Untouched & Safe) ---

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request=request, name="index.html", context={"history": online_chat_history, "sources": [], "mode": "medium", "engine": "online"})

@app.post("/search", response_class=HTMLResponse)
async def search(request: Request, query: str = Form(...), mode: str = Form(...), engine: str = Form(...), reset_history: str = Form("false")):
    if reset_history == "true": online_chat_history.clear()
    
    if engine == "online":
        result = online_search(query=query, mode=mode)
    else:
        result = offline_rag_search(query=query, mode=mode)
        online_chat_history.append({"user": query, "ai": result["answer"]})
        if len(online_chat_history) > 5: online_chat_history.pop(0)

    return templates.TemplateResponse(request=request, name="index.html", context={"history": online_chat_history, "sources": result.get("sources", []), "mode": mode, "engine": engine})

# --- PHASE 2: ADVANCED READER ROUTES ---

@app.get("/reader", response_class=HTMLResponse)
async def reader_page(request: Request):
    # Abhi humne reader.html banaya nahi hai, par route ready kar rahe hain
    return templates.TemplateResponse(request=request, name="reader.html", context={})

@app.get("/api/sermon_list")
async def get_sermon_list():
    # Sidebar index ke liye saare sermons ki list
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT code, year, title FROM sermons ORDER BY year ASC, code ASC")
    rows = cursor.fetchall()
    conn.close()
    return JSONResponse(content=[dict(ix) for ix in rows])

@app.get("/api/sermon/{code}")
async def get_sermon(code: str):
    # Pura sermon text load karne ke liye
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT title, code, year, text FROM sermons WHERE code = ?", (code,))
    row = cursor.fetchone()
    conn.close()
    if row:
        return JSONResponse(content=dict(row))
    return JSONResponse(content={"error": "Sermon not found"}, status_code=404)
