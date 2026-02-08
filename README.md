# Personal Shopper AI 🛒

Um assistente de compras conversacional que ajuda você a encontrar e comprar produtos.

**URL:** https://personal-shopper-production-e1bb.up.railway.app/

## Visão do Produto

Interface de chat onde o usuário conversa naturalmente e o AI encontra produtos, mostra reviews, e monta uma lista de compras - tudo numa conversa.

## Roadmap

### Fase 1: Fundação ✅
- [x] API de busca com Brave Search
- [x] Análise de produtos com Claude
- [x] Links diretos (não listas) do ML, Amazon, etc
- [x] Deploy no Railway

### Fase 2: Chat Interface 🚧
- [ ] Migrar para AI SDK (Vercel)
- [ ] Interface de chat conversacional
- [ ] Search como tool do AI
- [ ] Histórico de conversa

### Fase 3: Produtos Completos
- [ ] Imagens dos produtos
- [ ] Reviews YouTube integrados
- [ ] Reviews TikTok
- [ ] Preços em tempo real
- [ ] Melhorar cobertura de links (ML fraco)

### Fase 4: Lista de Compras
- [ ] Adicionar produtos à lista
- [ ] Comparar produtos lado a lado
- [ ] Salvar/compartilhar lista
- [ ] Checkout unificado (links)

### Fase 5: Polish
- [ ] UI mobile-friendly
- [ ] Cache para buscas repetidas
- [ ] Alertas de preço
- [ ] Autenticação/histórico

## Tech Stack

- **Backend:** FastAPI (Python)
- **AI:** Claude (Anthropic)
- **Search:** Brave Search API
- **Frontend:** HTML/JS (migrar para Next.js + AI SDK)
- **Deploy:** Railway

## Problemas Conhecidos

1. Tempo de busca lento (~22-30s)
2. Alguns produtos sem links de compra
3. Links do Mercado Livre escassos
4. Reviews YouTube inconsistentes

## Desenvolvimento

```bash
cd api
pip install -r requirements.txt
uvicorn main:app --reload
```

Variáveis de ambiente:
- `BRAVE_API_KEY`
- `ANTHROPIC_API_KEY`
