"""Chat endpoint - conversational shopping assistant"""
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional
import anthropic
import json
import os

from .search import find_product_links, find_youtube_reviews, find_product_image, brave_search

router = APIRouter()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

class Message(BaseModel):
    role: str  # "user" or "assistant"
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]
    language: str = "pt-BR"

# Define the search tool for Claude
SEARCH_TOOL = {
    "name": "search_products",
    "description": "Search for products to buy. Use this when the user asks about products, recommendations, or wants to buy something. Returns product recommendations with buy links, prices, reviews and images.",
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The product search query, e.g. 'fone bluetooth até 500 reais' or 'monitor 27 polegadas para programar'"
            },
            "max_price": {
                "type": "number",
                "description": "Maximum price in BRL (optional)"
            }
        },
        "required": ["query"]
    }
}

async def execute_search_tool(query: str, max_price: Optional[float] = None) -> dict:
    """Execute the product search and return formatted results"""
    try:
        # Get initial search
        price_q = f"até R${max_price}" if max_price else ""
        search_results = await brave_search(f"{query} {price_q} comprar", count=15)
        
        # Use Claude to extract product recommendations
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        
        results_text = "\n".join([
            f"{i+1}. {r.get('title', '')}\n   {r.get('url', '')}\n   {r.get('description', '')}"
            for i, r in enumerate(search_results.get("web", {}).get("results", [])[:12])
        ])
        
        resp = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1000,
            messages=[{
                "role": "user",
                "content": f"""Extract 3-4 specific product names from these search results. Return ONLY a JSON array of product names.

Search: {query}
Results:
{results_text}

Return format: ["Product Name 1", "Product Name 2", "Product Name 3"]"""
            }]
        )
        
        # Parse product names
        import re
        text = resp.content[0].text
        match = re.search(r'\[.*\]', text, re.DOTALL)
        if not match:
            return {"error": "No products found", "products": []}
        
        product_names = json.loads(match.group())
        
        # Enrich each product
        products = []
        for name in product_names[:4]:
            links = await find_product_links(name)
            if not links:
                continue
            
            reviews = await find_youtube_reviews(name)
            image = await find_product_image(name, links)
            
            # Get price from first link
            price = links[0].get("price") or "Ver preço"
            
            products.append({
                "name": name,
                "price": price,
                "buy_links": links,
                "reviews": reviews,
                "image_url": image
            })
        
        return {
            "query": query,
            "products": products,
            "count": len(products)
        }
    except Exception as e:
        return {"error": str(e), "products": []}

def format_products_for_display(result: dict) -> str:
    """Format product results for the assistant's response"""
    if not result.get("products"):
        return "Não encontrei produtos para essa busca. Tente ser mais específico."
    
    lines = [f"Encontrei {len(result['products'])} opções:\n"]
    
    for i, p in enumerate(result["products"], 1):
        lines.append(f"**{i}. {p['name']}**")
        lines.append(f"   💰 {p['price']}")
        if p.get("buy_links"):
            link = p["buy_links"][0]
            lines.append(f"   🛒 [{link['store']}]({link['url']})")
        if p.get("reviews"):
            review = p["reviews"][0]
            lines.append(f"   📺 [Review]({review['url']})")
        lines.append("")
    
    return "\n".join(lines)

@router.post("/chat")
async def chat(request: ChatRequest):
    """Conversational shopping assistant endpoint"""
    if not ANTHROPIC_API_KEY:
        raise HTTPException(status_code=500, detail="API key not configured")
    
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    
    # Build messages
    messages = [{"role": m.role, "content": m.content} for m in request.messages]
    
    system_prompt = """Você é um personal shopper AI amigável e útil. Seu objetivo é ajudar o usuário a encontrar e comprar produtos.

REGRAS:
1. Seja conciso e direto
2. Use a ferramenta search_products quando o usuário perguntar sobre produtos
3. Após receber resultados, apresente-os de forma organizada com preços e links
4. Faça perguntas para entender melhor as necessidades (orçamento, preferências, uso)
5. Não invente produtos - use apenas o que a ferramenta retornar
6. Seja conversacional e natural

FORMATO:
- Use **negrito** para nomes de produtos
- Use emojis para organizar (💰 preço, 🛒 comprar, 📺 review)
- Inclua links clicáveis"""

    try:
        # Call Claude with tools
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1500,
            system=system_prompt,
            tools=[SEARCH_TOOL],
            messages=messages
        )
        
        # Handle tool use
        while response.stop_reason == "tool_use":
            # Find the tool use block
            tool_use = None
            text_content = ""
            for block in response.content:
                if block.type == "tool_use":
                    tool_use = block
                elif block.type == "text":
                    text_content = block.text
            
            if tool_use and tool_use.name == "search_products":
                # Execute search
                search_result = await execute_search_tool(
                    query=tool_use.input.get("query", ""),
                    max_price=tool_use.input.get("max_price")
                )
                
                # Format results
                formatted = format_products_for_display(search_result)
                
                # Continue conversation with tool result
                messages.append({"role": "assistant", "content": response.content})
                messages.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": tool_use.id,
                        "content": json.dumps(search_result, ensure_ascii=False)
                    }]
                })
                
                response = client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=1500,
                    system=system_prompt,
                    tools=[SEARCH_TOOL],
                    messages=messages
                )
            else:
                break
        
        # Extract final text response
        final_text = ""
        for block in response.content:
            if hasattr(block, "text"):
                final_text += block.text
        
        return {"response": final_text}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
