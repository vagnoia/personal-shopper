"""Chat endpoint - conversational shopping assistant with deep conversation"""
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
    "description": """Busca produtos para compra. 
    
IMPORTANTE: Só use esta ferramenta quando tiver informações SUFICIENTES sobre:
- O que o usuário quer (categoria/tipo de produto)
- Para que uso/contexto
- Faixa de preço (mesmo que aproximada)
- Alguma preferência de estilo/marca/característica

Se faltar alguma dessas informações, NÃO busque ainda - pergunte primeiro.""",
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Query de busca DETALHADA com todas as especificações coletadas. Ex: 'fone bluetooth over-ear para academia até R$300 com cancelamento de ruído' em vez de apenas 'fone bluetooth'"
            },
            "context": {
                "type": "string",
                "description": "Resumo do contexto coletado: uso, preferências, restrições"
            }
        },
        "required": ["query", "context"]
    }
}

SYSTEM_PROMPT = """Você é um personal shopper experiente e atencioso. Seu trabalho é ajudar pessoas a encontrar o produto PERFEITO para elas - não qualquer produto, mas O CERTO.

## SUA PERSONALIDADE
- Curioso e genuinamente interessado em ajudar
- Faz perguntas inteligentes que mostram expertise
- Nunca apressado - prefere entender bem antes de sugerir
- Conversa de forma natural, como um amigo que entende do assunto

## REGRAS DE OURO

### 1. NUNCA BUSQUE IMEDIATAMENTE
Quando alguém pede algo genérico ("quero um fone", "preciso de um notebook"), você NÃO busca direto.
Primeiro, entenda o contexto com 2-4 perguntas naturais (uma por vez).

### 2. PERGUNTAS ESSENCIAIS (adapte ao produto)
- **USO**: "Pra que você vai usar principalmente?" (trabalho, lazer, esporte, presente...)
- **CONTEXTO**: "Onde/quando vai usar mais?" (casa, rua, academia, viagem...)
- **ESTILO**: "Tem preferência de visual/marca?" (minimalista, gamer, premium, custo-benefício...)
- **BUDGET**: "Quanto tá pensando em investir?" (faixa, flexível ou fixo)
- **RESTRIÇÕES**: "Tem algo que não pode faltar ou que você odeia?" (bateria, peso, cor, material...)

### 3. QUANDO BUSCAR
Só chame search_products quando tiver:
✅ Tipo de produto claro
✅ Uso/contexto principal
✅ Faixa de preço (mesmo vaga: "barato", "bom custo-benefício", "o melhor")
✅ Pelo menos 1 preferência ou restrição

### 4. QUERY DE QUALIDADE
Quando buscar, monte uma query RICA com tudo que coletou:
❌ "notebook"
✅ "notebook para programação tela 14-15 polegadas até R$5000 leve para carregar"

❌ "sofá"
✅ "sofá 3 lugares retrátil tecido cinza ou bege para sala pequena até R$3000"

### 5. FORMATO DAS RESPOSTAS
- Perguntas: curtas, naturais, uma de cada vez
- Ao apresentar produtos: use **negrito** nos nomes, seja conciso
- Sempre explique POR QUE aquele produto combina com o que a pessoa disse

## EXEMPLOS

USER: "quero comprar um fone"
BOM: "Opa! Pra que você mais usa fone? Trabalho, música, academia...?"
RUIM: [buscar "fone bluetooth"]

USER: "um fone pra academia, treino pesado"
BOM: "Entendi! Fone pra treino precisa aguentar suor e ficar firme. Você prefere intra-auricular (vai dentro do ouvido) ou quer algo maior tipo headphone?"
RUIM: [buscar "fone academia"]

USER: "intra, que não caia de jeito nenhum"
BOM: "Perfeito, intra com encaixe firme. E quanto você tá pensando em gastar? Tem desde opções de R$100 até os top de R$500+."

USER: "até uns 250"
BOM: [AGORA SIM buscar: "fone bluetooth intra-auricular esportivo resistente suor encaixe firme até R$250"]

## LEMBRE-SE
Você é um ESPECIALISTA ajudando um amigo. Não tenha pressa. As melhores recomendações vêm de entender bem o que a pessoa precisa."""

async def execute_search(query: str, context: str = "") -> dict:
    """Execute product search and return results"""
    try:
        # Use the enriched query
        search_results = await brave_search(f"{query} comprar Brasil", count=15)
        
        # Extract product names with Claude
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        
        results_text = "\n".join([
            f"- {r.get('title', '')} | {r.get('description', '')[:100]}"
            for r in search_results.get("web", {}).get("results", [])[:10]
        ])
        
        # Include context in the extraction prompt for better filtering
        context_note = f"\nCONTEXTO DO USUÁRIO: {context}" if context else ""
        
        resp = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=500,
            messages=[{
                "role": "user",
                "content": f"""Extraia 3-4 nomes de produtos ESPECÍFICOS desses resultados que melhor atendem a busca.
{context_note}

Busca: {query}

Resultados:
{results_text}

REGRAS:
- Nomes EXATOS dos produtos (marca + modelo)
- Só produtos que existem e são vendidos no Brasil
- Priorize os que melhor atendem ao contexto

Retorne APENAS JSON: ["Produto 1 Marca Modelo", "Produto 2 Marca Modelo"]"""
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
    """Conversational shopping endpoint with deep conversation"""
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
                # Execute search with context
                query = tool_use.input.get("query", "")
                context = tool_use.input.get("context", "")
                result = await execute_search(query, context)
                products = result.get("products", [])
                
                # Format for Claude
                if products:
                    products_text = "\n".join([
                        f"- {p['name']}: {p['price']} ({p['buy_links'][0]['store']})"
                        for p in products
                    ])
                    tool_result = f"Encontrei {len(products)} produtos:\n{products_text}"
                else:
                    tool_result = "Não encontrei produtos para essa busca. Tente reformular ou ser mais específico."
                
                # Continue with tool result
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
        # Fallback without tools
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
                    query = tool_use.input.get('query', '')
                    yield f"data: {json.dumps({'type': 'searching', 'query': query})}\n\n"
                    
                    # Execute search with context
                    context = tool_use.input.get("context", "")
                    result = await execute_search(query, context)
                    products = result.get("products", [])
                    
                    # Format for Claude
                    if products:
                        products_text = "\n".join([
                            f"- {p['name']}: {p['price']} ({p['buy_links'][0]['store']})"
                            for p in products
                        ])
                        tool_result = f"Encontrei {len(products)} produtos:\n{products_text}"
                    else:
                        tool_result = "Não encontrei produtos para essa busca. Tente reformular."
                    
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
