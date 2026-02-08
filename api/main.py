"""Personal Shopper API - Main entry point"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routers import search, chat

app = FastAPI(
    title="Personal Shopper API",
    description="Assistente de compras conversacional",
    version="2.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(search.router, prefix="/api", tags=["search"])
app.include_router(chat.router, prefix="/api", tags=["chat"])

@app.get("/")
async def root():
    return {
        "name": "Personal Shopper API",
        "version": "2.0.0",
        "endpoints": {
            "search": "/api/search",
            "chat": "/api/chat",
            "chat_stream": "/api/chat/stream"
        }
    }

@app.get("/health")
async def health():
    return {"status": "ok"}
