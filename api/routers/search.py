"""Search endpoint - simplified robust version"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List
import httpx
import os
import json
import anthropic
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
    image_url: Optional[str] = None

class SearchResponse(BaseModel):
    query: str
    recommendations: List[ProductRecommendation]
    search_time_ms: int
    language: str

def is_valid_product_url(url: str) -> bool:
    """Check if URL is a direct product page"""
    if not url:
        return False
    # Block list patterns
    if any(p in url for p in ["lista.mercadolivre", "/s?", "/search?", "?k=", "/gp/browse", "/gp/bestsellers"]):
        return False
    # ML: accept /p/MLB, /p/MLA, MLB-, produto.mercadolivre
    if "mercadolivre" in url:
        return any(p in url for p in ["/p/MLB", "/p/MLA", "MLB-", "MLA-", "produto.mercadolivre"])
    # Amazon: must have /dp/
    if "amazon.com" in url:
        return "/dp/" in url
    # KaBuM: must have /produto/
    if "kabum" in url:
        return "/produto/" in url
    # Other stores: accept
    return True

def extract_store(url: str) -> str:
    """Extract store name from URL"""
    stores = [
        ("mercadolivre", "Mercado Livre"), ("amazon.com.br", "Amazon"), 
        ("kabum", "KaBuM!"), ("magazineluiza", "Magalu"), ("magalu", "Magalu"),
        ("americanas", "Americanas"), ("casasbahia", "Casas Bahia"),
        ("samsung.com", "Samsung"), ("apple.com", "Apple"), ("dell.com", "Dell"),
        ("motorola.com", "Motorola"), ("xiaomi", "Xiaomi"), ("mi.com", "Xiaomi"),
        ("jbl.com", "JBL"), ("sony.com", "Sony"),
    ]
    for pattern, name in stores:
        if pattern in url:
            return name
    match = re.search(r'https?://(?:www\.)?([^/]+)', url)
    return match.group(1).split('.')[0].title() if match else "Loja"

async def brave_search(query: str, count: int = 10) -> dict:
    """Search using Brave API"""
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            "https://api.search.brave.com/res/v1/web/search",
            headers={"X-Subscription-Token": BRAVE_API_KEY},
            params={"q": query, "count": count, "country": "BR"},
            timeout=15.0
        )
        resp.raise_for_status()
        return resp.json()

async def find_product_links(product_name: str) -> List[dict]:
    """Find buy links for a product"""
    links = []
    seen_stores = set()
    
    # Search multiple stores
    searches = [
        f'{product_name} site:mercadolivre.com.br',
        f'{product_name} site:amazon.com.br',
        f'{product_name} comprar preço',
    ]
    
    for search_query in searches:
        if len(links) >= 3:
            break
        try:
            data = await brave_search(search_query, count=5)
            for result in data.get("web", {}).get("results", []):
                url = result.get("url", "")
                if is_valid_product_url(url):
                    store = extract_store(url)
                    if store not in seen_stores:
                        # Extract price
                        price = None
                        match = re.search(r'R\$\s*[\d.,]+', result.get("description", ""))
                        if match:
                            price = match.group(0)
                        links.append({"store": store, "url": url, "price": price})
                        seen_stores.add(store)
                        if len(links) >= 4:
                            break
        except Exception:
            continue
    
    return links

async def find_youtube_reviews(product_name: str) -> List[dict]:
    """Find YouTube reviews for a product"""
    videos = []
    try:
        data = await brave_search(f'{product_name} review youtube', count=5)
        for result in data.get("web", {}).get("results", []):
            url = result.get("url", "")
            if "youtube.com/watch" in url:
                videos.append({
                    "title": result.get("title", ""),
                    "url": url,
                    "channel": None
                })
                if len(videos) >= 2:
                    break
    except Exception:
        pass
    return videos

async def extract_og_image(url: str) -> Optional[str]:
    """Extract og:image from a product page"""
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(url, timeout=8.0, follow_redirects=True, headers={
                "User-Agent": "Mozilla/5.0 (compatible; PersonalShopperBot/1.0)"
            })
            if resp.status_code == 200:
                html = resp.text[:50000]  # First 50KB
                # Look for og:image
                import re
                patterns = [
                    r'<meta[^>]*property=["\']og:image["\'][^>]*content=["\']([^"\']+)["\']',
                    r'<meta[^>]*content=["\']([^"\']+)["\'][^>]*property=["\']og:image["\']',
                    r'"image":\s*"(https?://[^"]+)"',  # JSON-LD
                ]
                for pattern in patterns:
                    match = re.search(pattern, html, re.IGNORECASE)
                    if match:
                        img = match.group(1)
                        # Skip logos and icons
                        if img and "logo" not in img.lower() and "icon" not in img.lower():
                            return img
    except Exception:
        pass
    return None

async def find_product_image(product_name: str, buy_links: List[dict]) -> Optional[str]:
    """Find product image - prefer og:image from buy links"""
    # Strategy 1: Extract og:image from the first buy link
    for link in buy_links[:2]:
        url = link.get("url", "")
        if url:
            img = await extract_og_image(url)
            if img:
                return img
    
    # Strategy 2: Fallback to Brave Images API
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                "https://api.search.brave.com/res/v1/images/search",
                headers={"X-Subscription-Token": BRAVE_API_KEY},
                params={"q": f"{product_name} produto oficial", "count": 3},
                timeout=8.0
            )
            if resp.status_code == 200:
                data = resp.json()
                for r in data.get("results", []):
                    img = r.get("thumbnail", {}).get("src") or r.get("properties", {}).get("url")
                    if img and "logo" not in img.lower():
                        return img
    except Exception:
        pass
    return None

async def get_recommendations(query: str, search_results: dict, max_price: Optional[float], language: str) -> List[dict]:
    """Get product recommendations from Claude"""
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    
    results_text = "\n".join([
        f"{i+1}. {r.get('title', '')}\n   {r.get('url', '')}\n   {r.get('description', '')}"
        for i, r in enumerate(search_results.get("web", {}).get("results", [])[:12])
    ])
    
    price_note = f"Orçamento: até R${max_price}." if max_price else ""
    
    prompt = f"""Você é um personal shopper. Recomende 4-5 produtos ESPECÍFICOS baseado na busca.

BUSCA: {query}
{price_note}

RESULTADOS:
{results_text}

REGRAS:
1. Nome EXATO do produto (ex: "JBL Tune 520BT", "Samsung Galaxy Buds2")
2. Produtos POPULARES fáceis de achar em lojas BR
3. Preço REAL quando disponível

JSON apenas (sem markdown):
{{"recommendations": [{{"rank": 1, "name": "Nome Exato", "price_range": "R$ 299", "description": "Desc", "pros": ["pro"], "cons": ["con"], "buy_links": [], "recommendation_reason": "Motivo"}}]}}"""

    resp = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1500,
        messages=[{"role": "user", "content": prompt}]
    )
    
    text = resp.content[0].text.strip()
    try:
        return json.loads(text).get("recommendations", [])
    except:
        match = re.search(r'\{[\s\S]*\}', text)
        if match:
            return json.loads(match.group()).get("recommendations", [])
        return []

@router.post("/search", response_model=SearchResponse)
async def search_products(request: SearchRequest):
    import time
    start = time.time()
    
    if not BRAVE_API_KEY or not ANTHROPIC_API_KEY:
        raise HTTPException(status_code=500, detail="API keys not configured")
    
    try:
        # Initial search
        price_q = f"até R${request.max_price}" if request.max_price else ""
        search_q = f"{request.query} {price_q} comprar"
        search_results = await brave_search(search_q, count=15)
        
        # Get recommendations
        recs = await get_recommendations(request.query, search_results, request.max_price, request.language)
        
        # Enrich each product
        enriched = []
        for rec in recs[:5]:
            name = rec.get("name", "")
            if not name:
                continue
                
            # Get links, reviews, image
            links = await find_product_links(name)
            
            # Skip if no links found
            if not links:
                continue
            
            reviews = await find_youtube_reviews(name)
            image = await find_product_image(name, links)
            
            # Use first link's price if no price in rec
            price = rec.get("price_range", "")
            if not price or price == "Preço não disponível":
                price = links[0].get("price") or "Ver preço"
            
            enriched.append(ProductRecommendation(
                rank=len(enriched) + 1,
                name=name,
                price_range=price,
                description=rec.get("description", ""),
                pros=rec.get("pros", []),
                cons=rec.get("cons", []),
                buy_links=[ProductLink(**l) for l in links],
                review_videos=[ReviewVideo(**v) for v in reviews],
                recommendation_reason=rec.get("recommendation_reason", ""),
                image_url=image
            ))
        
        if len(enriched) < 2:
            raise HTTPException(status_code=404, detail="Não encontramos produtos suficientes. Tente outra busca.")
        
        return SearchResponse(
            query=request.query,
            recommendations=enriched,
            search_time_ms=int((time.time() - start) * 1000),
            language=request.language
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
