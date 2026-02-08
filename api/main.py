"""Personal Shopper AI - FastAPI Backend"""
import os
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

app = FastAPI(title="Personal Shopper AI", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.get("/api/info")
async def api_info():
    return {"name": "Personal Shopper AI", "version": "0.1.0", "status": "online"}

@app.get("/")
async def serve_frontend():
    index_path = FRONTEND_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path, media_type="text/html")
    return HTMLResponse("<h1>Frontend not found</h1>")

try:
    from routers import search, chat
    app.include_router(search.router, prefix="/api", tags=["search"])
    app.include_router(chat.router, prefix="/api", tags=["chat"])
except Exception as e:
    @app.post("/api/search")
    async def search_fallback():
        return JSONResponse({"error": str(e)}, status_code=500)
    @app.post("/api/chat")
    async def chat_fallback():
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/{path:path}")
async def serve_static(path: str):
    if path.startswith("api"):
        return JSONResponse({"error": "not found"}, status_code=404)
    file_path = FRONTEND_DIR / path
    if file_path.exists() and file_path.is_file():
        return FileResponse(file_path)
    return FileResponse(FRONTEND_DIR / "index.html", media_type="text/html")
