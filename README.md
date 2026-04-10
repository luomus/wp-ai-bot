# WordPress AI Chatbot – Backend

Answers questions in **Finnish** using content fetched from
[info.laji.fi](https://info.laji.fi) WordPress pages plus external metadata
sources such as rs.laji.fi and tun.fi.

---

## Architecture

```
WordPress REST API
       │
   fetch_pages()          – paginated fetch, published pages only
       │
   fetch_external_term_sources() – fetch field metadata from rs.laji.fi and tun.fi
       │
   clean_html()           – strip tags, normalise whitespace
       │
   chunk_text()           – overlapping ~400-word chunks
       │
   create_embeddings()    – OpenAI text-embedding-3-small (batched)
       │
   FAISS index (memory)   – Inner Product / cosine similarity
       │
GET /ask?q=…
       │
   search()               – embed query → top-5 chunks
       │
   generate_answer()      – gpt-4o-mini, Finnish, context-only
       │
  JSON response           – { answer, sources }
```

---

## Quick start

### Option A – Docker Compose (local)

#### 1. Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (includes Docker Compose v2)
- An OpenAI API key

#### 2. Create your `.env` file

`
OPENAI_API_KEY=...
`

#### 3. Build and start

```powershell
docker compose up --build
```

The first run fetches all WordPress pages, embeds them (1–2 min), then serves
the API. Subsequent starts load from the `embeddings_cache` volume instantly.

#### 4. Query the API

```bash
curl "http://localhost:8000/ask?q=Mikä+on+kotka?"
```

- Use `wp-widget.html` to embed the chat UI in a WordPress custom HTML block.

## Option B - OpenShift (not local)

1. Login and select project:

```powershell
oc login <cluster-url> --token=<token>
oc project <your-project>
```

2. Process and apply the template:

```powershell
oc process -f openshift-template.yaml -p OPENAI_API_KEY=<your-key> | oc apply -f -
```

3. Verify resources:

```powershell
oc get all
oc get routes
```

## API reference

### `GET /ask?q=<question>`

Ask a question. The model answers in Finnish using only the WordPress content.

```bash
curl "http://localhost:8000/ask?q=Mikä+on+kotka?"
```

**Response**

```json
{
    "answer": "Kotka on Luomuksen ja Suomen Lajitietokeskuksen rakentama luonnontieteellisten näytekokoelmien hallintajärjestelmä, joka palvelee kaikkia eliöryhmiä, kudosnäytteitä, fossiileja sekä kasvitieteellisten puutarhojen eläviä kokoelmia. Järjestelmää on kehitetty vuodesta 2012 ja se on ollut tuotantokäytössä syksystä 2012 alkaen. Kotka on käytössä lähes kaikissa suomalaisissa luonnontieteellisissä museoissa ja se tarjoaa monia toiminnallisuuksia, kuten näytedatan tallennusta, hakua, lainojen hallintaa ja raportointia. Lisätietoja löytyy osoitteesta [Kotka-kokoelmienhallintajärjestelmä](https://info.laji.fi/etusivu/kotka-kokoelmienhallintajarjestelma/).",
    "sources": [
        {
            "title": "Kotka-kokoelmienhallintajärjestelmä",
            "url": "https://info.laji.fi/etusivu/kotka-kokoelmienhallintajarjestelma/"
        },
        {
            "title": "Collection Management System",
            "url": "https://info.laji.fi/en/frontpage/collection-management-system/"
        },
        {
            "title": "Presentations",
            "url": "https://info.laji.fi/en/frontpage/mission/presentations/"
        },
        {
            "title": "Palvelujen esittely",
            "url": "https://info.laji.fi/etusivu/palvelujen-esittely/"
        },
        {
            "title": "OGC API Features Overview",
            "url": "https://info.laji.fi/en/frontpage/spatial-data/spatial-data-services/ogc-api-instructions/"
        }
    ]
}
```

### `GET /health`

Returns index readiness and the number of indexed chunks.

```bash
curl http://localhost:8000/health
```

### `POST /refresh`

Force a full re-fetch and re-embedding (deletes the cache). Use this after the
WordPress content has changed significantly.

```bash
curl -X POST http://localhost:8000/refresh
```

---

## Configuration

All tuneable constants live at the top of `main.py`:

| Constant | Default | Description |
|---|---|---|
| `WP_API_BASE` | `https://info.laji.fi/wp-json/wp/v2/pages` | WordPress REST API URL |
| `CHUNK_SIZE_WORDS` | `400` | Target words per chunk |
| `CHUNK_OVERLAP_WORDS` | `50` | Word overlap between chunks |
| `TOP_K` | `5` | Chunks retrieved per query |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | OpenAI embedding model |
| `CHAT_MODEL` | `gpt-4.1-mini` | OpenAI chat model |
| `CACHE_DIR` | `.` (local) / `/app/cache` (Docker) | Directory for the cache file |

---

## Caching behaviour

- On startup, a SHA-256 fingerprint of all page titles + URLs is compared against the cached fingerprint.
- If they match, embeddings are loaded from disk — no API calls are made.
- If they differ (new/deleted/renamed pages), the full pipeline runs and the cache is refreshed.
- Use `POST /refresh` to force a rebuild at any time.

---

| `CACHE_DIR` | `.` (local) / `/app/cache` (Docker) | Directory for the cache file |

---

## Project structure

```
wp-ai-bot/
├── main.py                  # All application code
├── requirements.txt         # Python dependencies
├── Dockerfile               # Multi-stage Docker image
├── docker-compose.yml       # Compose stack (service + named volume)
├── .dockerignore            # Keeps the build context lean
├── .env.example             # API key template – commit this
├── .gitignore               # Excludes .env, cache, venvs
├── wp-widget.html           # WordPress embed widget
└── README.md                # This file
```

`embeddings_cache.pkl` is auto-generated and stored in the `embeddings_cache`
Docker named volume (or in the project root for local runs). It is safe to delete.

---
