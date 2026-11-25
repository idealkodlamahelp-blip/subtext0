"""Azerbaycan kitablari ucun oz axtaris indeksiniz (Meilisearch).

Qurasdirma:
    pip install meilisearch pypdf
    # Meilisearch server (Docker):
    docker run -d -p 7700:7700 -e MEILI_MASTER_KEY=MASTER_KEY getmeili/meilisearch:latest

Kitablari indekse yuklemek:
    1) PDF-leri bir qovluga qoyun (meselen ./pdfs)
    2) books.json yaradin:
       [
         {"file": "kitab1.pdf", "book_id": "kitab1", "title": "Kitabin adi",
          "author": "Muellif", "url": "https://menbe-sayt.az/kitab1"}
       ]
    3) python local_index.py ingest ./pdfs books.json

Axtaris testi:
    python local_index.py search "xatirladigim cumle"

book_finder.py avtomatik bu faylin search_local_index funksiyasini istifade edir.
"""

import json
import os
import sys

import meilisearch

MEILI_URL = os.getenv("MEILI_URL", "http://localhost:7700")
MEILI_KEY = os.getenv("MEILI_MASTER_KEY", "MASTER_KEY")
INDEX_NAME = "book_chunks"

CHUNK_WORDS = 120
CHUNK_OVERLAP = 30

_FOLD = str.maketrans(
    {
        "ə": "e", "Ə": "e", "ı": "i", "I": "i", "İ": "i",
        "ö": "o", "Ö": "o", "ü": "u", "Ü": "u", "ç": "c", "Ç": "c",
        "ş": "s", "Ş": "s", "ğ": "g", "Ğ": "g",
    }
)


def fold(text: str) -> str:
    """Azerbaycan herflerini ASCII-ye cevirir ki, 'e' yazilsa da 'e' tapilsin."""
    return (text or "").translate(_FOLD).lower()


def get_index():
    client = meilisearch.Client(MEILI_URL, MEILI_KEY)
    return client.index(INDEX_NAME)


def chunk_words(text: str):
    words = text.split()
    step = CHUNK_WORDS - CHUNK_OVERLAP
    for start in range(0, max(len(words), 1), step):
        piece = words[start : start + CHUNK_WORDS]
        if len(piece) >= 15:
            yield " ".join(piece)


def extract_pages(pdf_path: str):
    """(sehife_nomresi, metn) qaytarir. Skan olunmus PDF-lerde metn bos ola biler -> OCR lazimdir."""
    from pypdf import PdfReader

    reader = PdfReader(pdf_path)
    for i, page in enumerate(reader.pages, 1):
        try:
            yield i, page.extract_text() or ""
        except Exception:
            yield i, ""


def ingest(pdf_dir: str, meta_path: str) -> None:
    index = get_index()
    index.update_searchable_attributes(["text_folded"])
    index.update_displayed_attributes(
        ["id", "book_id", "title", "author", "url", "page", "text"]
    )

    with open(meta_path, encoding="utf-8") as f:
        books = json.load(f)

    for book in books:
        path = os.path.join(pdf_dir, book["file"])
        docs, empty_pages, total_pages = [], 0, 0
        n = 0
        for page_no, text in extract_pages(path):
            total_pages += 1
            if len(text.strip()) < 30:
                empty_pages += 1
                continue
            for piece in chunk_words(text):
                n += 1
                docs.append(
                    {
                        "id": f"{book['book_id']}_{n}",
                        "book_id": book["book_id"],
                        "title": book["title"],
                        "author": book.get("author", ""),
                        "url": book["url"],
                        "page": page_no,
                        "text": piece,
                        "text_folded": fold(piece),
                    }
                )
        if docs:
            index.add_documents(docs, primary_key="id")
        print(f"{book['title']}: {len(docs)} hisse yuklendi")
        if total_pages and empty_pages / total_pages > 0.5:
            print("  DIQQAT: sehifelerin cogu bosdur, bu PDF skandir. OCR lazimdir (ocrmypdf -l aze+rus).")


def search_local_index(query: str, max_results: int = 8) -> list[dict]:
    """book_finder.py-nin gozlediyen formatda netice qaytarir."""
    q = query.strip().strip('"')
    # Meilisearch sorgunu maksimum 10 sozle mehdudlasdirir
    q = " ".join(fold(q).split()[:10])
    if not q:
        return []
    try:
        res = get_index().search(q, {"limit": max_results, "matchingStrategy": "last"})
    except Exception as e:
        print(f"[local_index] xeta: {e}", file=sys.stderr)
        return []

    out = []
    for h in res.get("hits", []):
        url = h["url"]
        if url.lower().endswith(".pdf"):
            url = f"{url}#page={h['page']}"
        out.append(
            {
                "id": f"local:{h['id']}",
                "source": "Azerbaijan Library",
                "title": h["title"],
                "author": h.get("author", ""),
                "excerpt": h["text"],
                "url": url,
            }
        )
    return out


if __name__ == "__main__":
    if len(sys.argv) >= 4 and sys.argv[1] == "ingest":
        ingest(sys.argv[2], sys.argv[3])
    elif len(sys.argv) >= 3 and sys.argv[1] == "search":
        for r in search_local_index(" ".join(sys.argv[2:])):
            print(f"{r['title']} | {r['url']}\n   {r['excerpt'][:150]}...\n")
    else:
        print(__doc__)
