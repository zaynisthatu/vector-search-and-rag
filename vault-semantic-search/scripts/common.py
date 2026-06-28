"""Shared helpers for the VAULT semantic-search scripts.

Settings come from environment variables so the same code runs on Colab, locally, or in a test:
  VAULT_DB      path to the SQLite database with a `posts` table   (default: vault.db)
  QDRANT_URL    Qdrant server URL, or ":memory:" for an in-process instance (default: http://127.0.0.1:6333)
  EMBED_MODEL   sentence-transformers model name                   (default: BAAI/bge-base-en-v1.5)
  COLLECTION    Qdrant collection name                             (default: posts)
"""
import os
import sqlite3

DB_PATH = os.environ.get("VAULT_DB", "vault.db")
QDRANT_URL = os.environ.get("QDRANT_URL", "http://127.0.0.1:6333")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "BAAI/bge-base-en-v1.5")
COLLECTION = os.environ.get("COLLECTION", "posts")
VECTOR_SIZE = 768          # bge-base-en-v1.5 produces 768-dimensional vectors
COMMENT_CHARS = 300        # how much of the comments text goes into the embedded string

POST_COLUMNS = "post_id, username, caption, hashtags, comments_text, categories, music_title"


def get_client(url=None):
    from qdrant_client import QdrantClient
    url = url or QDRANT_URL
    if url == ":memory:":
        return QdrantClient(":memory:")
    return QdrantClient(url=url, check_compatibility=False)


def get_model(name=None, device=None):
    from sentence_transformers import SentenceTransformer
    if device is None:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return SentenceTransformer(name or EMBED_MODEL, device=device)


def load_posts(db_path=None):
    conn = sqlite3.connect(db_path or DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        return conn.execute(f"SELECT {POST_COLUMNS} FROM posts").fetchall()
    finally:
        conn.close()


def post_text(p):
    """The string that gets embedded: caption, hashtags, username, music title, start of the comments."""
    text = " ".join(filter(None, [
        p["caption"] or "", p["hashtags"] or "", p["username"] or "",
        p["music_title"] or "", (p["comments_text"] or "")[:COMMENT_CHARS],
    ]))
    return text.strip() or "no caption"


def payload(p):
    return {"post_id": p["post_id"], "username": p["username"],
            "caption": p["caption"] or "", "categories": p["categories"] or ""}
