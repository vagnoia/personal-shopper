"""Search endpoint - core functionality"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List
import httpx
import os
import json
import anthropic
import asyncio
import re

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
    image_url: Optional[str] = None  # NEW: product image

class SearchResponse(BaseModel):
    query: str
    recommendations: List[ProductRecommendation]
    search_time_ms: int
    language: str

def is_valid_product_url(url: str) -> bool:
    """Check if URL is a direct product page, not a list/search page"""
    if not url:
        return False
    invalid_patterns = [
        "lista.mercadolivre", "/s?", "/search", "?k=", "/b/",
        "mercadolivre.com.br/ofertas", "mercadolivre.com.br/c/",
        "/gp/browse/", "/gp/bestsellers/", "amazon.com.br/b/",
    ]
    if any(p in url for p in invalid_patterns):
        return False
    # Mercado Livre must have /p/MLB or /p/MLA or MLB- pattern in path
    if "mercadolivre" in url:
        if "/p/MLB" not in url and "/p/MLA" not in url and "MLB-" not in url:
            return False
    # Amazon must have /dp/ pattern
    if "amazon.com" in url and "/dp/" not in url:
        return False
    return True

def extract_store_name(url: str) -> str:
    """Extract store name from URL"""
    if "mercadolivre" in url:
        return "Mercado Livre"
    elif "amazon.com.br" in url:
        return "Amazon"
    elif "magazineluiza" in url or "magalu" in url:
        return "Magazine Luiza"
    elif "kabum" in url:
        return "KaBuM!"
    elif "americanas" in url:
        return "Americanas"
    elif "casasbahia" in url:
        return "Casas Bahia"
    elif "samsung.com" in url:
        return "Samsung"
    elif "apple.com" in url:
        return "Apple"
    elif "dell.com" in url:
        return "Dell"
    elif "motorola.com" in url:
        return "Motorola"
    elif "xiaomi" in url or "mi.com" in url:
        return "Xiaomi"
    elif "jbl.com" in url:
        return "JBL"
    elif "sony.com" in url:
        return "Sony"
    else:
        # Extract domain
        match = re.search(r'https?://(?:www\.)?([^/]+)', url)
        if match:
            return match.group(1).split('.')[0].title()
        return "Loja"

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

async def search_brave_images(query: str, count: int = 5) -> List[str]:
    """Search for product images"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://api.search.brave.com/res/v1/images/search",
                headers={"X-Subscription-Token": BRAVE_API_KEY},
                params={"q": query, "count": count, "safesearch": "moderate"},
                timeout=10.0
            )
            if response.status_code == 200:
                data = response.json()
                images = []
                for result in data.get("results", []):
                    img_url = result.get("properties", {}).get("url") or result.get("thumbnail", {}).get("src")
                    if img_url and not any(x in img_url for x in ["placeholder", "no-image", "default"]):
                        images.append(img_url)
                return images[:3]
    except:
        pass
    return []

async def search_product_links_multi(product_name: str) -> List[dict]:
    """Search for product links using multiple strategies"""
    links = []
    seen_stores = set()
    
    # Strategy 1: Direct store searches
    store_searches = [
        ("Mercado Livre", f'{product_name} site:mercadolivre.com.br/p/'),
        ("Amazon", f'{product_name} site:amazon.com.br/dp/'),
        ("KaBuM!", f'{product_name} site:kabum.com.br/produto/'),
        ("Magazine Luiza", f'{product_name} site:magazineluiza.com.br'),
    ]
    
    async def search_store(store_name: str, query: str) -> Optional[dict]:
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    "https://api.search.brave.com/res/v1/web/search",
                    headers={"X-Subscription-Token": BRAVE_API_KEY},
                    params={"q": query, "count": 3},
                    timeout=8.0
                )
                if response.status_code != 200:
                    return None
                data = response.json()
                
                for result in data.get("web", {}).get("results", []):
                    url = result.get("url", "")
                    if is_valid_product_url(url):
                        # Try to extract price from snippet
                        desc = result.get("description", "")
                        price = None
                        price_match = re.search(r'R\$\s*[\d.,]+', desc)
                        if price_match:
                            price = price_match.group(0)
                        return {"store": store_name, "url": url, "price": price}
        except:
            pass
        return None
    
    # Run store searches in parallel
    tasks = [search_store(store, query) for store, query in store_searches]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    for result in results:
        if result and isinstance(result, dict) and result.get("store") not in seen_stores:
            links.append(result)
            seen_stores.add(result["store"])
    
    # Strategy 2: Generic search for "comprar [product]" 
    if len(links) < 2:
        try:
            data = await search_brave(f"{product_name} comprar preço", count=10)
            for result in data.get("web", {}).get("results", []):
                url = result.get("url", "")
                if is_valid_product_url(url):
                    store = extract_store_name(url)
                    if store not in seen_stores:
                        desc = result.get("description", "")
                        price = None
                        price_match = re.search(r'R\$\s*[\d.,]+', desc)
                        if price_match:
                            price = price_match.group(0)
                        links.append({"store": store, "url": url, "price": price})
                        seen_stores.add(store)
                        if len(links) >= 4:
                            break
        except:
            pass
    
    return links

async def search_youtube_reviews(product_name: str, language: str = "pt-BR") -> List[dict]:
    """Search for YouTube reviews - always try to find at least one"""
    videos = []
    queries = [
        f"{product_name} review",
        f"{product_name} análise",
        f"{product_name} vale a pena",
        f"{product_name} unboxing",
    ] if language == "pt-BR" else [
        f"{product_name} review",
        f"{product_name} hands on",
    ]
    
    for query in queries:
        if len(videos) >= 2:
            break
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    "https://api.search.brave.com/res/v1/web/search",
                    headers={"X-Subscription-Token": BRAVE_API_KEY},
                    params={"q": f"{query} youtube", "count": 5},
                    timeout=8.0
                )
                if response.status_code != 200:
                    continue
                data = response.json()
                
                for result in data.get("web", {}).get("results", []):
                    url = result.get("url", "")
                    if "youtube.com/watch" in url and url not in [v["url"] for v in videos]:
                        title = result.get("title", "")
                        # Extract channel from title if possible
                        channel = None
                        if " - " in title:
                            channel = title.split(" - ")[-1].strip()
                        videos.append({
                            "title": title,
                            "url": url,
                            "channel": channel
                        })
                        if len(videos) >= 2:
                            break
        except:
            continue
    
    return videos

async def analyze_with_claude(query: str, search_results: dict, max_price: Optional[float], language: str) -> List[dict]:
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    
    results_text = ""
    for i, result in enumerate(search_results.get("web", {}).get("results", [])[:15]):
        results_text += f"\n{i+1}. {result.get('title', '')}\n   URL: {result.get('url', '')}\n   {result.get('description', '')}\n"
    
    price_instruction = f"O usuário tem orçamento máximo de R${max_price}." if max_price else ""
    lang_instruction = "Responda em português brasileiro." if language == "pt-BR" else "Respond in English."
    
    prompt = f"""Você é um personal shopper AI especialista. Analise os resultados e recomende 3-5 produtos ESPECÍFICOS e POPULARES.

BUSCA DO USUÁRIO: {query}
{price_instruction}

RESULTADOS DA PESQUISA:
{results_text}

REGRAS CRÍTICAS:
1. Recomende APENAS produtos ESPECÍFICOS com nome e modelo exato (ex: "Samsung Galaxy Buds2 Pro", "JBL Tune 520BT")
2. Escolha produtos POPULARES e CONHECIDOS que são fáceis de encontrar em lojas brasileiras
3. Os buy_links são opcionais aqui - vamos enriquecer depois
4. Inclua preço REAL quando disponível (ex: "R$ 899", não "R$ 500 - R$ 1000")
5. NUNCA invente produtos que não existem

{lang_instruction}

Retorne APENAS JSON válido (sem markdown):
{{"recommendations": [{{"rank": 1, "name": "Nome Exato do Produto", "price_range": "R$ 899", "description": "Descrição concisa", "pros": ["pro1", "pro2"], "cons": ["con1"], "buy_links": [], "recommendation_reason": "Por que esse produto é bom para o usuário"}}]}}"""

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
        json_match = re.search(r'\{[\s\S]*\}', response_text)
        if json_match:
            data = json.loads(json_match.group())
            return data.get("recommendations", [])
        return []

async def enrich_product(rec: dict, language: str) -> Optional[ProductRecommendation]:
    """Enrich a product with links, reviews, and images. Returns None if no links found."""
    product_name = rec.get("name", "")
    
    # Parallel fetch: links, reviews, images
    links_task = search_product_links_multi(product_name)
    reviews_task = search_youtube_reviews(product_name, language)
    images_task = search_brave_images(f"{product_name} produto")
    
    links, reviews, images = await asyncio.gather(links_task, reviews_task, images_task)
    
    # CRITICAL: Skip products without any links
    if not links:
        return None
    
    # Get best image
    image_url = images[0] if images else None
    
    # Build the enriched product
    return ProductRecommendation(
        rank=rec.get("rank", 0),
        name=product_name,
        price_range=rec.get("price_range", links[0].get("price") or "Ver preço no site"),
        description=rec.get("description", ""),
        pros=rec.get("pros", []),
        cons=rec.get("cons", []),
        buy_links=[ProductLink(**link) for link in links],
        review_videos=[ReviewVideo(**vid) for vid in reviews],
        recommendation_reason=rec.get("recommendation_reason", ""),
        image_url=image_url
    )

@router.post("/search", response_model=SearchResponse)
async def search_products(request: SearchRequest):
    import time
    start_time = time.time()
    
    if not BRAVE_API_KEY:
        raise HTTPException(status_code=500, detail="Brave API key not configured")
    if not ANTHROPIC_API_KEY:
        raise HTTPException(status_code=500, detail="Anthropic API key not configured")
    
    try:
        # Build search query
        price_query = f"até R${request.max_price}" if request.max_price and request.language == "pt-BR" else ""
        search_query = f"{request.query} {price_query} comprar" if request.language == "pt-BR" else f"{request.query} {price_query} buy"
        
        # Get initial search results
        search_results = await search_brave(search_query, request.country)
        
        # Get Claude's recommendations
        recommendations = await analyze_with_claude(request.query, search_results, request.max_price, request.language)
        
        # Enrich products in parallel
        enrich_tasks = [enrich_product(rec, request.language) for rec in recommendations[:6]]
        enriched_results = await asyncio.gather(*enrich_tasks, return_exceptions=True)
        
        # Filter out None (products without links) and errors
        enriched = []
        for i, result in enumerate(enriched_results):
            if result and isinstance(result, ProductRecommendation):
                result.rank = len(enriched) + 1  # Re-rank
                enriched.append(result)
        
        # If we got less than 3 products, the search quality is low
        if len(enriched) < 2:
            raise HTTPException(
                status_code=404, 
                detail="Não encontramos produtos suficientes com links de compra. Tente uma busca mais específica."
            )
        
        return SearchResponse(
            query=request.query,
            recommendations=enriched[:5],
            search_time_ms=int((time.time() - start_time) * 1000),
            language=request.language
        )
    except HTTPException:
        raise
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Search service error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")
