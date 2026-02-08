"""Search endpoint - core functionality"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List
import httpx
import os
import json
import anthropic

router = APIRouter()

BRAVE_API_KEY = os.getenv("BRAVE_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=500)
    max_price: Optional[float] = None
    language: str = "pt-BR"
    country: str = "BR"

class ProductLink(BaseModel):
    store: str
    url: str
    price: Optional[str] = None

class ReviewVideo(BaseModel):
    title: str
    url: str
    channel: Optional[str] = None

class ProductRecommendation(BaseModel):
    rank: int
    name: str
    price_range: str
    description: str
    pros: List[str]
    cons: List[str]
    buy_links: List[ProductLink]
    review_videos: List[ReviewVideo]
    recommendation_reason: str

class SearchResponse(BaseModel):
    query: str
    recommendations: List[ProductRecommendation]
    search_time_ms: int
    language: str

async def search_brave(query: str, country: str = "BR", count: int = 20) -> dict:
    async with httpx.AsyncClient() as client:
        response = await client.get(
            "https://api.search.brave.com/res/v1/web/search",
            headers={"X-Subscription-Token": BRAVE_API_KEY},
            params={"q": query, "country": country, "count": count},
            timeout=15.0
        )
        response.raise_for_status()
        return response.json()

async def search_product_links(product_name: str) -> List[dict]:
    """Search for specific product buy links from major stores"""
    links = []
    stores = [
        ("Mercado Livre", "site:mercadolivre.com.br"),
        ("Amazon", "site:amazon.com.br"),
        ("Magazine Luiza", "site:magazineluiza.com.br"),
    ]
    
    try:
        # Search each store
        for store_name, site_filter in stores[:2]:  # Limit to 2 stores to avoid rate limits
            try:
                query = f'{product_name} {site_filter}'
                async with httpx.AsyncClient() as client:
                    response = await client.get(
                        "https://api.search.brave.com/res/v1/web/search",
                        headers={"X-Subscription-Token": BRAVE_API_KEY},
                        params={"q": query, "count": 3},
                        timeout=8.0
                    )
                    if response.status_code != 200:
                        continue
                    data = response.json()
                
                for result in data.get("web", {}).get("results", []):
                    url = result.get("url", "")
                    title = result.get("title", "")
                    # Filter for product pages (not list/search pages)
                    if any(x in url for x in ["/p/MLB", "/dp/", "/produto/", "MLB-", "-MLB"]) and \
                       not any(x in url for x in ["lista.", "/s?", "/search", "?k="]):
                        links.append({"store": store_name, "url": url, "price": None})
                        break  # One link per store
            except:
                continue
    except:
        pass
    
    return links

async def search_youtube_reviews(product_name: str, language: str = "pt-BR") -> List[dict]:
    try:
        lang_query = "review português" if language == "pt-BR" else "review"
        query = f"{product_name} {lang_query} youtube"
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://api.search.brave.com/res/v1/web/search",
                headers={"X-Subscription-Token": BRAVE_API_KEY},
                params={"q": query, "count": 5},
                timeout=10.0
            )
            response.raise_for_status()
            data = response.json()
        videos = []
        for result in data.get("web", {}).get("results", []):
            if "youtube.com/watch" in result.get("url", ""):
                videos.append({"title": result.get("title", ""), "url": result.get("url", ""), "channel": None})
        return videos[:3]
    except Exception:
        return []

async def analyze_with_claude(query: str, search_results: dict, max_price: Optional[float], language: str) -> List[dict]:
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    
    results_text = ""
    for i, result in enumerate(search_results.get("web", {}).get("results", [])[:15]):
        results_text += f"\n{i+1}. {result.get('title', '')}\n   URL: {result.get('url', '')}\n   {result.get('description', '')}\n"
    
    price_instruction = f"O usuário tem orçamento máximo de R${max_price}." if max_price else ""
    lang_instruction = "Responda em português brasileiro." if language == "pt-BR" else "Respond in English."
    
    prompt = f"""Você é um personal shopper AI especialista. Analise os resultados e recomende 3-5 produtos ESPECÍFICOS.

BUSCA DO USUÁRIO: {query}
{price_instruction}

RESULTADOS DA PESQUISA:
{results_text}

REGRAS IMPORTANTES:
1. Recomende produtos ESPECÍFICOS com nome e modelo exato (ex: "JBL Tune 520BT", não "Fone JBL")
2. Os buy_links DEVEM ser URLs diretas para a PÁGINA DO PRODUTO, não páginas de busca ou listas
3. URLs válidas: amazon.com.br/dp/XXXXX, mercadolivre.com.br/MLB-XXXXX, loja.com/produto/nome
4. URLs INVÁLIDAS (não use): amazon.com.br/s?k=..., lista.mercadolivre.com.br/..., /search?q=...
5. Se não encontrar link direto do produto, use o link mais específico disponível nos resultados
6. Inclua preço real quando disponível nos resultados

{lang_instruction}

Retorne APENAS JSON válido (sem markdown):
{{"recommendations": [{{"rank": 1, "name": "Modelo Específico do Produto", "price_range": "R$ X - R$ Y", "description": "Descrição", "pros": ["pro1", "pro2"], "cons": ["con1"], "buy_links": [{{"store": "Amazon", "url": "https://www.amazon.com.br/dp/XXXXXXX", "price": "R$ 199"}}], "recommendation_reason": "Motivo"}}]}}"""

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}]
    )
    
    response_text = response.content[0].text.strip()
    try:
        data = json.loads(response_text)
        return data.get("recommendations", [])
    except json.JSONDecodeError:
        import re
        json_match = re.search(r'\{[\s\S]*\}', response_text)
        if json_match:
            data = json.loads(json_match.group())
            return data.get("recommendations", [])
        return []

@router.post("/search", response_model=SearchResponse)
async def search_products(request: SearchRequest):
    import time
    start_time = time.time()
    
    if not BRAVE_API_KEY:
        raise HTTPException(status_code=500, detail="Brave API key not configured")
    if not ANTHROPIC_API_KEY:
        raise HTTPException(status_code=500, detail="Anthropic API key not configured")
    
    try:
        price_query = f"até R${request.max_price}" if request.max_price and request.language == "pt-BR" else ""
        search_query = f"{request.query} {price_query} comprar" if request.language == "pt-BR" else f"{request.query} {price_query} buy"
        
        search_results = await search_brave(search_query, request.country)
        recommendations = await analyze_with_claude(request.query, search_results, request.max_price, request.language)
        
        enriched = []
        for rec in recommendations[:5]:
            # Enrich with real product links
            try:
                real_links = await search_product_links(rec.get("name", ""))
                if real_links:
                    rec["buy_links"] = real_links + rec.get("buy_links", [])[:1]
            except:
                pass
            
            # Enrich with YouTube reviews
            try:
                videos = await search_youtube_reviews(rec.get("name", ""), request.language)
                rec["review_videos"] = videos if videos else rec.get("review_videos", [])
            except:
                rec["review_videos"] = rec.get("review_videos", [])
            
            enriched.append(ProductRecommendation(
                rank=rec.get("rank", len(enriched) + 1),
                name=rec.get("name", "Unknown"),
                price_range=rec.get("price_range", "Preço não disponível"),
                description=rec.get("description", ""),
                pros=rec.get("pros", []),
                cons=rec.get("cons", []),
                buy_links=[ProductLink(**link) for link in rec.get("buy_links", [])],
                review_videos=[ReviewVideo(**vid) for vid in rec.get("review_videos", [])],
                recommendation_reason=rec.get("recommendation_reason", "")
            ))
        
        return SearchResponse(
            query=request.query,
            recommendations=enriched,
            search_time_ms=int((time.time() - start_time) * 1000),
            language=request.language
        )
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Search service error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")
