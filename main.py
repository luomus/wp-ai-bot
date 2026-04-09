"""
WordPress AI Chatbot Backend
Answers questions using ONLY content fetched from WordPress pages via REST API.
"""

import os
import json
import hashlib
import pickle
import re
import time
import logging
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
load_dotenv()  # Load GEMINI_API_KEY from .env if present

import requests
import faiss
import numpy as np
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import openai
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

WP_API_BASE = "https://info.laji.fi/wp-json/wp/v2/pages"
# Honour CACHE_DIR env var so Docker can mount a persistent volume there
_cache_dir = Path(os.environ.get("CACHE_DIR", "."))
_cache_dir.mkdir(parents=True, exist_ok=True)
CACHE_FILE = _cache_dir / "embeddings_cache.pkl"
CHUNK_SIZE_WORDS = 400       # Target words per chunk
CHUNK_OVERLAP_WORDS = 50     # Overlap between consecutive chunks
TOP_K = int(os.environ.get("TOP_K", "5"))  # Number of chunks to retrieve per query (env configurable)
MIN_SCORE = float(os.environ.get("MIN_SCORE", "0.2"))  # Minimal similarity score for strong hits
EMBEDDING_MODEL = "text-embedding-3-small"  # OpenAI embedding model
CHAT_MODEL = "gpt-4o-mini"

# Configure OpenAI API for embeddings and chat
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise EnvironmentError("OPENAI_API_KEY environment variable is not set.")
openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

class AnswerResponse(BaseModel):
    answer: str
    sources: list[dict]

# ---------------------------------------------------------------------------
# Step 1 – Fetch WordPress pages
# ---------------------------------------------------------------------------

def fetch_pages() -> list[dict]:
    """
    Fetch ALL pages from the WordPress REST API, handling pagination.
    Returns a list of dicts with keys: title, content, url.
    """
    pages = []
    url = WP_API_BASE
    params = {"per_page": 100, "page": 1, "_fields": "id,title,content,link,status"}

    while url:
        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
        except requests.RequestException as exc:
            logger.error("Failed to fetch page %s: %s", params.get("page"), exc)
            break

        data = resp.json()
        if not data:
            break

        for page in data:
            # Only include published pages
            if page.get("status") != "publish":
                continue
            pages.append({
                "title": page["title"]["rendered"],
                "content": page["content"]["rendered"],
                "url": page["link"],
            })

        # WordPress returns total pages in the header
        total_pages = int(resp.headers.get("X-WP-TotalPages", 1))
        current_page = params["page"]
        if current_page >= total_pages:
            break

        params = {"per_page": 100, "page": current_page + 1, "_fields": "id,title,content,link,status"}

    logger.info("Fetched %d published pages from WordPress.", len(pages))
    return pages

# ---------------------------------------------------------------------------
# Step 2 – Clean HTML content
# ---------------------------------------------------------------------------

def clean_html(html: str) -> str:
    """
    Strip all HTML tags, scripts, styles, and normalise whitespace.
    Returns plain text.
    """
    soup = BeautifulSoup(html, "html.parser")

    # Remove script and style elements entirely
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    text = soup.get_text(separator=" ")

    # Collapse multiple whitespace characters into a single space
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ---------------------------------------------------------------------------
# Step 3 – Chunk text
# ---------------------------------------------------------------------------

def chunk_text(text: str, title: str, url: str) -> list[dict]:
    """
    Split text into overlapping word-based chunks of ~CHUNK_SIZE_WORDS words.
    Each chunk retains the page title and URL as metadata.
    """
    words = text.split()
    if not words:
        return []

    chunks = []
    start = 0

    while start < len(words):
        end = min(start + CHUNK_SIZE_WORDS, len(words))
        chunk_words = words[start:end]
        chunk_text_str = " ".join(chunk_words)

        if chunk_text_str.strip():
            chunks.append({
                "title": title,
                "url": url,
                "text": chunk_text_str,
            })

        # Move forward by (chunk_size - overlap) so consecutive chunks share context
        step = CHUNK_SIZE_WORDS - CHUNK_OVERLAP_WORDS
        start += step

    return chunks

# ---------------------------------------------------------------------------
# Step 4 – Create embeddings
# ---------------------------------------------------------------------------

def create_embeddings(chunks: list[dict]) -> tuple[np.ndarray, list[dict]]:
    """
    Embed all chunk texts using the OpenAI Embeddings API.
    Returns a float32 numpy array (n_chunks × embedding_dim) and the chunks list.

    Sends texts in batches to respect API limits and reduce latency.
    """
    BATCH_SIZE = 100
    all_embeddings: list[list[float]] = []
    texts = [chunk["text"] for chunk in chunks]

    for batch_start in range(0, len(texts), BATCH_SIZE):
        batch = texts[batch_start: batch_start + BATCH_SIZE]
        logger.info(
            "Embedding batch %d–%d / %d …",
            batch_start + 1,
            batch_start + len(batch),
            len(texts),
        )
        try:
            response = openai_client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        except Exception as exc:
            logger.error("OpenAI embedding error: %s", exc)
            raise

        # OpenAI v1 returns response.data as a list of Embedding objects
        for item in response.data:
            all_embeddings.append(item.embedding)

    matrix = np.array(all_embeddings, dtype=np.float32)
    logger.info("Created %d embeddings of dimension %d.", matrix.shape[0], matrix.shape[1])
    return matrix, chunks

# ---------------------------------------------------------------------------
# Step 5 – Build / load FAISS index
# ---------------------------------------------------------------------------

def build_faiss_index(matrix: np.ndarray) -> faiss.IndexFlatIP:
    """
    Build an in-memory FAISS index using Inner Product (cosine similarity after
    L2-normalisation).
    """
    # Normalise vectors so inner product equals cosine similarity
    faiss.normalize_L2(matrix)
    dim = matrix.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(matrix)
    logger.info("FAISS index built with %d vectors.", index.ntotal)
    return index

# ---------------------------------------------------------------------------
# Step 5b – Cache helpers
# ---------------------------------------------------------------------------

def _pages_fingerprint(pages: list[dict]) -> str:
    """Compute a stable hash of page titles + URLs to detect stale caches."""
    key = json.dumps(
        [{"title": p["title"], "url": p["url"]} for p in pages],
        ensure_ascii=False,
    )
    return hashlib.sha256(key.encode()).hexdigest()

def load_cache() -> Optional[dict]:
    """Return the cached payload dict or None if no valid cache exists."""
    if CACHE_FILE.exists():
        try:
            with CACHE_FILE.open("rb") as fh:
                return pickle.load(fh)
        except Exception as exc:
            logger.warning("Could not read cache file: %s", exc)
    return None

def save_cache(payload: dict) -> None:
    """Persist the payload dict to disk."""
    try:
        with CACHE_FILE.open("wb") as fh:
            pickle.dump(payload, fh)
        logger.info("Cache saved to %s.", CACHE_FILE)
    except Exception as exc:
        logger.warning("Could not write cache file: %s", exc)

# ---------------------------------------------------------------------------
# Step 6 – Semantic search
# ---------------------------------------------------------------------------

def search(query: str, index: faiss.IndexFlatIP, chunks: list[dict], top_k: int = TOP_K) -> list[dict]:
    """
    Embed the user query and return the top_k most similar chunks.
    """
    try:
        resp = openai_client.embeddings.create(model=EMBEDDING_MODEL, input=query)
    except Exception as exc:
        logger.error("OpenAI embedding error for query: %s", exc)
        raise HTTPException(status_code=502, detail="Embedding API error.")

    # Extract embedding from OpenAI v1 response
    embedding = resp.data[0].embedding
    query_vec = np.array([embedding], dtype=np.float32)
    faiss.normalize_L2(query_vec)

    distances, indices = index.search(query_vec, top_k)

    logger.debug("Search query=%s top_k=%d distances=%s indices=%s", query, top_k, distances[0].tolist(), indices[0].tolist())

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx == -1:
            continue
        chunk = chunks[idx].copy()
        chunk["score"] = float(dist)
        results.append(chunk)

    if results:
        best_score = results[0]["score"]
        if best_score < MIN_SCORE:
            logger.warning("Weak semantic match: best_score=%.4f for query='%s'", best_score, query)

    return results

# ---------------------------------------------------------------------------
# Step 7 – Generate answer via LLM
# ---------------------------------------------------------------------------

def generate_answer(query: str, relevant_chunks: list[dict]) -> AnswerResponse:
    """
    Build a prompt from the retrieved chunks and call the Gemini Chat API.
    The model is instructed to:
      - answer in Finnish
      - use ONLY the provided context
      - say "En tiedä" if the answer is not in the context
    """
    if not relevant_chunks:
        return AnswerResponse(answer="En tiedä", sources=[])

    # Build context block
    context_parts = []
    for i, chunk in enumerate(relevant_chunks, 1):
        context_parts.append(
            f"[{i}] Sivu: {chunk['title']}\nURL: {chunk['url']}\n\n{chunk['text']}"
        )
    context = "\n\n---\n\n".join(context_parts)

    system_prompt = (
        "Olet avulias assistentti, joka vastaa kysymyksiin AINOASTAAN alla olevan kontekstin perusteella. "
        "Vastaa aina suomeksi. "
        "Jos et löydä vastausta annetusta kontekstista, sano täsmälleen: 'En tiedä, mutta ehkä seuraavat sivut auttavat: '. "
        "Älä keksi tietoja, joita ei ole kontekstissa. "
        "Viittaa lähteisiin luonnollisesti vastauksessasi."
    )

    # Combine system prompt with user message
    try:
        completion = openai_client.chat.completions.create(
            model=CHAT_MODEL,
            temperature=0.2,
            max_tokens=1024,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Konteksti:\n\n{context}\n\nKysymys: {query}"},
            ],
        )
    except Exception as exc:
        logger.error("OpenAI chat error: %s", exc)
        raise HTTPException(status_code=502, detail="Chat API error.")

    answer_text = completion.choices[0].message.content.strip()

    # Deduplicate sources, preserving order
    seen = set()
    sources = []
    for chunk in relevant_chunks:
        key = chunk["url"]
        if key not in seen:
            seen.add(key)
            sources.append({"title": chunk["title"], "url": chunk["url"]})

    return AnswerResponse(answer=answer_text, sources=sources)

# ---------------------------------------------------------------------------
# Application startup – load / build index
# ---------------------------------------------------------------------------

# Rate limiting
limiter = Limiter(key_func=get_remote_address, default_limits=["20/hour"])

app = FastAPI(
    title="WordPress AI Chatbot",
    description="Answers questions based solely on WordPress page content.",
    version="1.0.0",
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, lambda request, exc: HTTPException(status_code=429, detail="Rate limit exceeded. Max 5 requests per minute."))

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

# Global state initialised at startup
_faiss_index: Optional[faiss.IndexFlatIP] = None
_chunks: list[dict] = []

@app.on_event("startup")
async def startup_event() -> None:
    """
    On startup:
      1. Fetch all WordPress pages.
      2. Try to load embeddings from the cache; rebuild if pages have changed.
      3. Build the FAISS index into memory.
    """
    global _faiss_index, _chunks

    logger.info("Fetching WordPress pages …")
    pages = fetch_pages()
    if not pages:
        logger.warning("No pages fetched; the /ask endpoint will return empty answers.")
        return

    fingerprint = _pages_fingerprint(pages)
    cache = load_cache()

    if cache and cache.get("fingerprint") == fingerprint:
        logger.info("Cache is up-to-date. Loading embeddings from disk.")
        _chunks = cache["chunks"]
        matrix = cache["matrix"]
    else:
        logger.info("Cache miss or stale. Building embeddings from scratch …")

        # Clean and chunk all pages
        all_chunks: list[dict] = []
        for page in pages:
            plain_text = clean_html(page["content"])
            page_chunks = chunk_text(plain_text, page["title"], page["url"])
            all_chunks.extend(page_chunks)

        logger.info("Total chunks to embed: %d", len(all_chunks))

        matrix, _chunks = create_embeddings(all_chunks)

        save_cache({"fingerprint": fingerprint, "chunks": _chunks, "matrix": matrix})

    _faiss_index = build_faiss_index(matrix)
    logger.info("Startup complete. Ready to answer questions.")

# ---------------------------------------------------------------------------
# API endpoint
# ---------------------------------------------------------------------------

@app.get("/ask", response_model=AnswerResponse)
@limiter.limit("5/minute")
async def ask(request: Request, q: str = Query(..., min_length=2, description="Question to ask the chatbot")) -> AnswerResponse:
    """
    Embed the query, retrieve the most relevant WordPress page chunks,
    and return an LLM-generated answer in Finnish together with source URLs.
    """
    if _faiss_index is None or not _chunks:
        raise HTTPException(
            status_code=503,
            detail="Index not ready. WordPress pages may not have been fetched yet.",
        )

    relevant_chunks = search(q, _faiss_index, _chunks)
    return generate_answer(q, relevant_chunks)


@app.get("/health")
async def health() -> dict:
    """Quick health-check endpoint."""
    return {
        "status": "ok",
        "chunks_indexed": len(_chunks),
        "index_ready": _faiss_index is not None,
    }


@app.post("/refresh")
async def refresh() -> dict:
    """
    Force a full re-fetch and re-embedding of all WordPress pages.
    Deletes the cache file so the next startup (or this call) rebuilds everything.
    """
    global _faiss_index, _chunks

    if CACHE_FILE.exists():
        CACHE_FILE.unlink()
        logger.info("Cache file deleted.")

    logger.info("Re-fetching pages and rebuilding index …")
    pages = fetch_pages()
    if not pages:
        raise HTTPException(status_code=502, detail="Could not fetch any pages.")

    all_chunks: list[dict] = []
    for page in pages:
        plain_text = clean_html(page["content"])
        page_chunks = chunk_text(plain_text, page["title"], page["url"])
        all_chunks.extend(page_chunks)

    matrix, _chunks = create_embeddings(all_chunks)
    save_cache({
        "fingerprint": _pages_fingerprint(pages),
        "chunks": _chunks,
        "matrix": matrix,
    })
    _faiss_index = build_faiss_index(matrix)

    return {"status": "refreshed", "chunks_indexed": len(_chunks)}
