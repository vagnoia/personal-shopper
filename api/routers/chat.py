"""Chat endpoint - conversational shopping assistant"""
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, AsyncGenerator
import anthropic
import json
import os
import re
import asyncio

from .search import find_product_links, find_youtube_reviews, find_product_image, brave_search

router = APIRouter()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]
    language: str = "pt-BR"

# Tool definition
SEARCH_TOOL = {
    "name": "search_products",
    "description": "Busca produtos para compra. Use quando o usuário perguntar sobre produtos, pedir recomendações ou quiser comprar algo.",
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Busca do produto, ex: 'fone bluetooth até 500 reais'"
            }
        },
        "required": ["query"]
    }
}

SYSTEM_PROMPT = """Você é um personal shopper. Ajuda pessoas a encontrar produtos.

ANTES DE BUSCAR, considere se precisa saber:
- Para qual uso? (academia, trabalho, presente, dia a dia...)
- Preferência de marca ou característica?
- Faixa de preço?

Use seu julgamento: se a pergunta já tiver contexto suficiente, busque direto. Se faltar algo importante, faça UMA pergunta curta e natural.

FORMATO:
- Perguntas: curtas e diretas, 1 frase
- Respostas com produtos: breves, nomes em **negrito**"""

async def execute_search(query: str) -> dict:
    """Execute product search and return results"""
    try:
        # Search
        search_results = await brave_search(f"{query} comprar", count=15)
        
        # Extract product names with Claude
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        
        results_text = "\n".join([
            f"- {r.get('title', '')} | {r.get('description', '')[:100]}"
            for r in search_results.get("web", {}).get("results", [])[:10]
        ])
        
        resp = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=500,
            messages=[{
                "role": "user",
                "content": f"Extraia 3-4 nomes de produtos específicos desses resultados.\nBusca: {query}\nResultados:\n{results_text}\n\nRetorne APENAS JSON: [\"Produto 1\", \"Produto 2\"]"
            }]
        )
        
        # Parse products
        match = re.search(r'\[.*\]', resp.content[0].text, re.DOTALL)
        if not match:
            return {"products": [], "error": "No products found"}
        
        product_names = json.loads(match.group())
        
        # Enrich products
        products = []
        for name in product_names[:4]:
            links = await find_product_links(name)
            if not links:
                continue
            
            reviews = await find_youtube_reviews(name)
            image = await find_product_image(name, links)
            price = links[0].get("price") or "Ver preço"
            
            products.append({
                "name": name,
                "price": price,
                "buy_links": links,
                "review_videos": reviews,
                "image_url": image
            })
        
        return {"products": products, "query": query}
    
    except Exception as e:
        return {"products": [], "error": str(e)}

@router.post("/chat")
async def chat(request: ChatRequest):
    """Conversational shopping endpoint"""
    if not ANTHROPIC_API_KEY:
        raise HTTPException(status_code=500, detail="API not configured")
    
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    messages = [{"role": m.role, "content": m.content} for m in request.messages]
    
    try:
        # Call Claude with tool
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            tools=[SEARCH_TOOL],
            messages=messages
        )
        
        products = None
        
        # Handle tool use
        if response.stop_reason == "tool_use":
            tool_use = next((b for b in response.content if b.type == "tool_use"), None)
            
            if tool_use and tool_use.name == "search_products":
                # Execute search
                result = await execute_search(tool_use.input.get("query", ""))
                products = result.get("products", [])
                
                # Format for Claude
                if products:
                    products_text = "\n".join([
                        f"- {p['name']}: {p['price']} ({p['buy_links'][0]['store']})"
                        for p in products
                    ])
                    tool_result = f"Encontrei {len(products)} produtos:\n{products_text}"
                else:
                    tool_result = "Não encontrei produtos para essa busca."
                
                # Continue with tool result - serialize content blocks to plain dicts
                assistant_content = []
                for block in response.content:
                    if block.type == "text":
                        assistant_content.append({"type": "text", "text": block.text})
                    elif block.type == "tool_use":
                        assistant_content.append({
                            "type": "tool_use",
                            "id": block.id,
                            "name": block.name,
                            "input": block.input
                        })
                messages.append({"role": "assistant", "content": assistant_content})
                messages.append({
                    "role": "user", 
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": tool_use.id,
                        "content": tool_result
                    }]
                })
                
                response = client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=1024,
                    system=SYSTEM_PROMPT,
                    tools=[SEARCH_TOOL],
                    messages=messages
                )
        
        # Extract text
        text = ""
        for block in response.content:
            if hasattr(block, "text"):
                text += block.text
        
        return {"response": text, "products": products}
    
    except anthropic.BadRequestError as e:
        # Fallback without tools for older API
        try:
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1024,
                system=SYSTEM_PROMPT,
                messages=messages
            )
            return {"response": response.content[0].text, "products": None}
        except Exception as e2:
            raise HTTPException(status_code=500, detail=str(e2))
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """Streaming conversational shopping endpoint using SSE"""
    if not ANTHROPIC_API_KEY:
        raise HTTPException(status_code=500, detail="API not configured")
    
    async def generate() -> AsyncGenerator[str, None]:
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        messages = [{"role": m.role, "content": m.content} for m in request.messages]
        products = None
        
        try:
            # First call with streaming
            with client.messages.stream(
                model="claude-sonnet-4-20250514",
                max_tokens=1024,
                system=SYSTEM_PROMPT,
                tools=[SEARCH_TOOL],
                messages=messages
            ) as stream:
                full_response = None
                collected_text = ""
                
                for event in stream:
                    if hasattr(event, 'type'):
                        if event.type == 'content_block_delta':
                            if hasattr(event.delta, 'text'):
                                collected_text += event.delta.text
                                yield f"data: {json.dumps({'type': 'text', 'content': event.delta.text})}\n\n"
                        elif event.type == 'message_stop':
                            full_response = stream.get_final_message()
            
            if not full_response:
                yield f"data: {json.dumps({'type': 'error', 'content': 'No response'})}\n\n"
                return
            
            # Check if tool was called
            if full_response.stop_reason == "tool_use":
                tool_use = next((b for b in full_response.content if b.type == "tool_use"), None)
                
                if tool_use and tool_use.name == "search_products":
                    # Notify frontend that we're searching
                    yield f"data: {json.dumps({'type': 'searching', 'query': tool_use.input.get('query', '')})}\n\n"
                    
                    # Execute search
                    result = await execute_search(tool_use.input.get("query", ""))
                    products = result.get("products", [])
                    
                    # Format for Claude
                    if products:
                        products_text = "\n".join([
                            f"- {p['name']}: {p['price']} ({p['buy_links'][0]['store']})"
                            for p in products
                        ])
                        tool_result = f"Encontrei {len(products)} produtos:\n{products_text}"
                    else:
                        tool_result = "Não encontrei produtos para essa busca."
                    
                    # Serialize content blocks
                    assistant_content = []
                    for block in full_response.content:
                        if block.type == "text":
                            assistant_content.append({"type": "text", "text": block.text})
                        elif block.type == "tool_use":
                            assistant_content.append({
                                "type": "tool_use",
                                "id": block.id,
                                "name": block.name,
                                "input": block.input
                            })
                    
                    messages.append({"role": "assistant", "content": assistant_content})
                    messages.append({
                        "role": "user",
                        "content": [{
                            "type": "tool_result",
                            "tool_use_id": tool_use.id,
                            "content": tool_result
                        }]
                    })
                    
                    # Stream the final response
                    with client.messages.stream(
                        model="claude-sonnet-4-20250514",
                        max_tokens=1024,
                        system=SYSTEM_PROMPT,
                        tools=[SEARCH_TOOL],
                        messages=messages
                    ) as stream2:
                        for event in stream2:
                            if hasattr(event, 'type') and event.type == 'content_block_delta':
                                if hasattr(event.delta, 'text'):
                                    yield f"data: {json.dumps({'type': 'text', 'content': event.delta.text})}\n\n"
            
            # Send products at the end
            if products:
                yield f"data: {json.dumps({'type': 'products', 'content': products})}\n\n"
            
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
            
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )
