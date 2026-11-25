"""Book Finder: user remembers a fragment/sentence from a book -> real book found.

Pipeline:
  1. LLM analyzes the query (never invents book titles — only prepares search queries)
  2. Real search: Google Books + Open Library (swap in your own index below)
  3. LLM picks among candidates only (it cannot invent)
  4. Code builds the link from candidate id (LLM never emits URLs)

Setup:
    pip install openai requests
    export OPENAI_API_KEY="sk-..."
    # optional: export GOOGLE_BOOKS_API_KEY="..."

Run:
    python book_finder.py "the quote you remember"
"""

import json
import os
import re
import sys
from typing import Any

import requests
from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
GOOGLE_BOOKS_API_KEY = os.getenv("GOOGLE_BOOKS_API_KEY")  # optional
MAX_CANDIDATES = 15
MIN_CONFIDENCE = 0.3
HTTP_TIMEOUT = 15

client = OpenAI()  # reads OPENAI_API_KEY from env

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
QUERY_SYSTEM_PROMPT = """You are a query-understanding component of a book search system.

The user remembers a fragment, sentence, or idea from a book and wants to find the exact book. Your job is NOT to name the book. Your job is to prepare search queries for a real search engine.

Rules:
- Never guess or invent book titles, authors, or URLs.
- The user's text may be in Azerbaijani, Turkish, Russian, English, or mixed. It may be an inexact, paraphrased, or translated memory of the original text.
- Azerbaijani text may contain missing diacritics (e.g. "e" instead of "ə", "i" instead of "ı", "s" instead of "ş", "g" instead of "ğ", "o" instead of "ö", "u" instead of "ü", "c" instead of "ç"). Restore the likely correct spelling.
- Generate multiple search variants to maximize recall.
- Exact phrase queries must be SHORT (3-8 words), because long phrases rarely match verbatim.

Return ONLY valid JSON in this exact format, no extra text:

{
  "detected_language": "az | tr | ru | en | other",
  "normalized_quote": "the quote with corrected spelling, same language",
  "is_likely_translation": true,
  "exact_phrase_queries": ["short exact phrases likely to appear verbatim"],
  "keyword_queries": ["important keywords and named entities, space separated"],
  "semantic_queries": ["1-2 sentence description of the meaning of the quote"],
  "translated_queries": {
    "en": ["English translation of the quote"],
    "az": ["Azerbaijani version"],
    "ru": ["Russian version"],
    "tr": ["Turkish version"]
  },
  "named_entities": ["people, places, or unusual terms mentioned"],
  "notes": "any ambiguity worth flagging"
}"""

VERIFY_SYSTEM_PROMPT = """You are a verification component of a book search system.

You receive:
1. The user's remembered quote.
2. A list of candidate passages returned by a real search engine. Each has an id, source, book title, author, and a text excerpt.

Your task: decide which candidates (if any) actually match the user's quote.

Strict rules:
- Use ONLY the provided candidates. Never add books, authors, or facts from your own memory.
- Never output URLs. The system will build links from the candidate id.
- A match requires clear overlap in wording or very specific meaning. Vague thematic similarity is NOT a match.
- If the user's quote is a paraphrase or translation, you may still match, but lower the confidence and say so.
- If a candidate has an empty or very short excerpt, you cannot verify it; give it at most 0.3 confidence.
- If no candidate matches well, return an empty matches list. Do not force a match.
- Confidence scale: 0.9-1.0 = near-exact wording; 0.6-0.89 = strong paraphrase or translation; 0.3-0.59 = weak/possible; below 0.3 = do not include.
- Sort matches by confidence, highest first.

Return ONLY valid JSON:

{
  "matches": [
    {
      "candidate_id": "id exactly as provided",
      "confidence": 0.0,
      "match_type": "exact | close_paraphrase | translation | weak",
      "evidence": "quote the matching part of the excerpt, and explain in one sentence why it matches"
    }
  ],
  "no_confident_match": false,
  "suggestion": "if no confident match, suggest how the user could refine their memory (e.g. add character names, approximate genre, language)"
}"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def strip_html(text: str) -> str:
    """Clean HTML tags and entities in Google Books snippets."""
    text = re.sub(r"<[^>]+>", "", text or "")
    text = (
        text.replace("&#39;", "'")
        .replace("&quot;", '"')
        .replace("&amp;", "&")
        .replace("&nbsp;", " ")
    )
    return re.sub(r"\s+", " ", text).strip()


def chat_json(system_prompt: str, user_message: str) -> dict[str, Any]:
    """Call the LLM and return JSON."""
    resp = client.chat.completions.create(
        model=MODEL,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
    )
    return json.loads(resp.choices[0].message.content)


# ---------------------------------------------------------------------------
# Stage 1: Analyze the query
# ---------------------------------------------------------------------------
def analyze_query(user_text: str) -> dict[str, Any]:
    user_message = (
        "User memory of the book fragment:\n"
        '"""\n' + user_text + '\n"""'
    )
    return chat_json(QUERY_SYSTEM_PROMPT, user_message)


def build_search_strings(analysis: dict[str, Any], original: str) -> list[str]:
    """Collect search queries from the LLM analysis."""
    queries: list[str] = []

    # 1) Exact phrases (quoted — best for Google Books)
    for q in analysis.get("exact_phrase_queries", []):
        if q and q.strip():
            queries.append(f'"{q.strip()}"')

    # 2) Normalized quote (short version)
    norm = (analysis.get("normalized_quote") or original).strip()
    words = norm.split()
    if len(words) > 8:
        queries.append(f'"{" ".join(words[:8])}"')
    elif words:
        queries.append(f'"{norm}"')

    # 3) Keyword queries
    for q in analysis.get("keyword_queries", []):
        if q and q.strip():
            queries.append(q.strip())

    # 4) Translated variants
    for lang_queries in (analysis.get("translated_queries") or {}).values():
        for q in lang_queries or []:
            w = (q or "").split()
            if w:
                queries.append(f'"{" ".join(w[:8])}"')

    # Deduplicate, preserve order
    seen, unique = set(), []
    for q in queries:
        key = q.lower()
        if key not in seen:
            seen.add(key)
            unique.append(q)
    return unique[:10]


# ---------------------------------------------------------------------------
# Stage 2: Real search sources
# Every function returns the same shape:
#   {"id", "source", "title", "author", "excerpt", "url"}
# ---------------------------------------------------------------------------
def search_google_books(query: str, max_results: int = 8) -> list[dict]:
    params = {"q": query, "maxResults": max_results, "printType": "books"}
    if GOOGLE_BOOKS_API_KEY:
        params["key"] = GOOGLE_BOOKS_API_KEY
    try:
        r = requests.get(
            "https://www.googleapis.com/books/v1/volumes",
            params=params,
            timeout=HTTP_TIMEOUT,
        )
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"[google_books] error: {e}", file=sys.stderr)
        return []

    results = []
    for item in data.get("items", []):
        info = item.get("volumeInfo", {})
        snippet = strip_html(item.get("searchInfo", {}).get("textSnippet", ""))
        description = strip_html(info.get("description", ""))[:300]
        vol_id = item.get("id")
        if not vol_id:
            continue
        results.append(
            {
                "id": f"gb:{vol_id}",
                "source": "Google Books",
                "title": info.get("title", ""),
                "author": ", ".join(info.get("authors", [])),
                "excerpt": snippet or description,
                "url": info.get("infoLink")
                or f"https://books.google.com/books?id={vol_id}",
            }
        )
    return results


def search_open_library(query: str, max_results: int = 8) -> list[dict]:
    """Open Library search-inside (full-text search in digitized books)."""
    try:
        r = requests.get(
            "https://openlibrary.org/search/inside.json",
            params={"q": query, "limit": max_results},
            timeout=HTTP_TIMEOUT,
        )
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"[open_library] error: {e}", file=sys.stderr)
        return []

    results = []
    for hit in (data.get("hits") or {}).get("hits", []):
        edition = hit.get("edition") or {}
        fields = hit.get("fields") or {}
        ia_list = fields.get("ia") or fields.get("meta_ia") or []
        ia = ia_list[0] if isinstance(ia_list, list) and ia_list else None
        key = edition.get("key") if isinstance(edition, dict) else None

        highlight = (hit.get("highlight") or {}).get("text") or []
        excerpt = strip_html(" ... ".join(highlight)).replace("{{{", "").replace("}}}", "")

        if key:
            url = f"https://openlibrary.org{key}"
            cid = f"ol:{key}"
        elif ia:
            url = f"https://archive.org/details/{ia}"
            cid = f"ia:{ia}"
        else:
            continue

        title = ""
        author = ""
        if isinstance(edition, dict):
            title = edition.get("title", "")
            authors = edition.get("authors") or []
            author = ", ".join(a.get("name", "") for a in authors if isinstance(a, dict))

        results.append(
            {
                "id": cid,
                "source": "Open Library",
                "title": title,
                "author": author,
                "excerpt": excerpt,
                "url": url,
            }
        )
    return results


def search_local_index(query: str, max_results: int = 8) -> list[dict]:
    """Fill this in with YOUR search (Azerbaijan library, Elasticsearch, etc.).

    Example with Meilisearch:

        import meilisearch
        ms = meilisearch.Client("http://localhost:7700", "MASTER_KEY")
        res = ms.index("book_chunks").search(query, {"limit": max_results})
        return [{
            "id": f"local:{h['book_id']}",
            "source": "Azerbaijan Library",
            "title": h["title"],
            "author": h["author"],
            "excerpt": h["text"],
            "url": f"https://YOUR-SITE.az/book/{h['book_id']}",
        } for h in res["hits"]]
    """
    return []


def collect_candidates(queries: list[str]) -> list[dict]:
    """Run all searches, dedupe by id."""
    seen_ids: set[str] = set()
    candidates: list[dict] = []

    for q in queries:
        for search_fn in (search_local_index, search_google_books, search_open_library):
            for c in search_fn(q):
                if c["id"] not in seen_ids:
                    seen_ids.add(c["id"])
                    candidates.append(c)

    # Drop very short excerpts (the LLM cannot verify those)
    candidates = [c for c in candidates if len(c.get("excerpt", "")) >= 20]
    return candidates[:MAX_CANDIDATES]


# ---------------------------------------------------------------------------
# Stage 3: Verify candidates
# ---------------------------------------------------------------------------
def verify_candidates(user_text: str, candidates: list[dict]) -> dict[str, Any]:
    if not candidates:
        return {
            "matches": [],
            "no_confident_match": True,
            "suggestion": "Axtaris heç bir namized tapmadi. Personaj adlari, janr, "
            "or cemlənin başqa tutsağını elave edin.",
        }

    # Never send URLs to the LLM — only data it needs for verification
    llm_candidates = [
        {
            "id": c["id"],
            "source": c["source"],
            "title": c["title"],
            "author": c["author"],
            "excerpt": c["excerpt"],
        }
        for c in candidates
    ]

    user_message = (
        "User's remembered quote:\n"
        '"""\n' + user_text + '\n"""\n\n'
        "Candidates:\n" + json.dumps(llm_candidates, ensure_ascii=False, indent=2)
    )
    return chat_json(VERIFY_SYSTEM_PROMPT, user_message)


# ---------------------------------------------------------------------------
# Stage 4: Final output — links built from candidate ids only
# ---------------------------------------------------------------------------
def find_book(user_text: str, verbose: bool = True) -> dict[str, Any]:
    user_text = user_text.strip()
    if not user_text:
        return {"results": [], "message": "Bos sorgu."}

    # 1) Analysis
    analysis = analyze_query(user_text)
    queries = build_search_strings(analysis, user_text)
    if verbose:
        print("Axtaris sorgulari:")
        for q in queries:
            print("  -", q)

    # 2) Search
    candidates = collect_candidates(queries)
    if verbose:
        print(f"\nTapilan namized sayi: {len(candidates)}")

    # 3) Verification
    verdict = verify_candidates(user_text, candidates)

    # 4) Final: only real candidates with confidence above the threshold
    by_id = {c["id"]: c for c in candidates}
    results = []
    for m in verdict.get("matches", []):
        cand = by_id.get(m.get("candidate_id"))
        try:
            conf = float(m.get("confidence", 0))
        except (TypeError, ValueError):
            conf = 0.0
        if cand is None or conf < MIN_CONFIDENCE:
            continue
        results.append(
            {
                "title": cand["title"],
                "author": cand["author"],
                "source": cand["source"],
                "url": cand["url"],  # built by code from the candidate id
                "confidence": conf,
                "match_type": m.get("match_type"),
                "evidence": m.get("evidence"),
            }
        )

    results.sort(key=lambda r: r["confidence"], reverse=True)

    return {
        "results": results,
        "no_confident_match": len(results) == 0,
        "suggestion": verdict.get("suggestion", ""),
        "debug": {"analysis": analysis, "queries": queries},
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    if len(sys.argv) < 2:
        print('Istifade: python book_finder.py "xatirladigim cumle"')
        sys.exit(1)

    out = find_book(" ".join(sys.argv[1:]))

    print("\n=== NETICE ===")
    if out["no_confident_match"]:
        print("Deqiq uygunluq tapilmadi.")
        if out["suggestion"]:
            print("Meslehet:", out["suggestion"])
        return

    for i, r in enumerate(out["results"], 1):
        print(f"\n{i}. {r['title']}  -  {r['author']}")
        print(f"   Menbe: {r['source']}   Eminlik: {r['confidence']:.2f} ({r['match_type']})")
        print(f"   Link: {r['url']}")
        print(f"   Delil: {r['evidence']}")


if __name__ == "__main__":
    main()
