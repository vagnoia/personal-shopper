"""Chat endpoint - conversational shopping assistant"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import anthropic
import json
import os
import re

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

SYSTEM_PROMPT = """Você é um personal shopper simpático e prestativo. Seu trabalho é ajudar pessoas a encontrar e comprar produtos.

PERSONALIDADE:
- Seja natural e amigável, como um amigo que entende de compras
- Use linguagem casual mas profissional
- Faça perguntas para entender melhor as necessidades
- Dê opiniões e recomendações baseadas no que sabe

QUANDO BUSCAR PRODUTOS:
- Quando o usuário pedir recomendação de produto
- Quando mencionar que quer comprar algo
- Quando perguntar sobre preços ou opções

QUANDO NÃO BUSCAR:
- Quando fizer perguntas sobre produtos já mostrados
- Quando pedir comparação entre produtos já listados
- Quando só estiver conversando

FORMATO:
- Seja conciso mas útil
- Destaque nomes de produtos em **negrito**
- Quando mostrar produtos, comente brevemente sobre cada um
- Pergunte se quer mais detalhes ou outras opções"""

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
