from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from web.ai_engine import offline_rag_search
from web.hybrid_engine import online_search, hybrid_search, online_chat_history

app = FastAPI()
app.mount("/static", StaticFiles(directory="web/static"), name="static")
templates = Jinja2Templates(directory="web/templates")

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
        # Offline result ko history mein yahan add karenge
        online_chat_history.append({"user": query, "ai": result["answer"]})
        if len(online_chat_history) > 5: online_chat_history.pop(0)

    return templates.TemplateResponse(request=request, name="index.html", context={"history": online_chat_history, "sources": result.get("sources", []), "mode": mode, "engine": engine})
