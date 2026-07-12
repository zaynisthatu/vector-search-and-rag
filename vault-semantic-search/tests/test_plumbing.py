"""Plumbing test for the indexing and search scripts.

Uses an in-memory Qdrant and a fake encoder (deterministic vectors from a hash of the text), so it checks
the code paths: build, incremental indexing, pagination, search and payloads. It does NOT test the quality
of the real embedding model.

    python tests/test_plumbing.py
"""
import hashlib
import os
import sqlite3
import sys
import tempfile

import numpy as np
from qdrant_client import QdrantClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import common                                   # noqa: E402
from build_index import build_index             # noqa: E402
from index_new_posts import index_new_posts, existing_state   # noqa: E402
from search import search                       # noqa: E402


class FakeModel:
    """Same interface as SentenceTransformer.encode for what the scripts use."""
    def encode(self, texts, batch_size=32, show_progress_bar=False, normalize_embeddings=True):
        out = []
        for t in texts:
            seed = int(hashlib.sha256(t.encode()).hexdigest()[:8], 16)
            v = np.random.default_rng(seed).normal(size=common.VECTOR_SIZE)
            out.append(v / np.linalg.norm(v))
        return np.array(out)


def make_db(path, n):
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE posts (post_id TEXT, username TEXT, caption TEXT, hashtags TEXT, "
                 "comments_text TEXT, categories TEXT, music_title TEXT)")
    rows = [(f"p{i}", f"user{i % 5}", f"caption number {i} about topic {i % 7}", "#demo", "nice", "misc", None)
            for i in range(n)]
    rows.append(("pnull", None, None, None, None, None, None))   # a post with every field empty
    conn.executemany("INSERT INTO posts VALUES (?,?,?,?,?,?,?)", rows)
    conn.commit(); conn.close()


def main():
    tmp = tempfile.mkdtemp()
    db = os.path.join(tmp, "vault.db")
    client, model = QdrantClient(":memory:"), FakeModel()

    make_db(db, 10)
    assert build_index(client, model, db_path=db) == 11
    info = client.get_collection("posts")
    assert info.points_count == 11
    assert info.config.params.vectors.size == 768
    assert info.config.params.vectors.distance.name == "COSINE"

    # nothing new -> nothing added
    assert index_new_posts(client, model, db_path=db) == 0

    # 2 new posts -> exactly 2 added, with fresh ids
    conn = sqlite3.connect(db)
    conn.execute("INSERT INTO posts VALUES ('new1','a','a brand new caption','#x','c','misc',NULL)")
    conn.execute("INSERT INTO posts VALUES ('new2','b','another new caption','#y','c','misc',NULL)")
    conn.commit(); conn.close()
    assert index_new_posts(client, model, db_path=db) == 2
    ids, max_id = existing_state(client, "posts", page=3)       # small page size forces pagination
    assert len(ids) == 13 and max_id == 12, (len(ids), max_id)

    # the exact text of a post is its own nearest neighbour with score ~1.0
    posts = {p["post_id"]: p for p in common.load_posts(db)}
    q = common.post_text(posts["p3"])
    top = search(client, model, q, limit=3)
    assert top[0][1]["post_id"] == "p3" and top[0][0] > 0.999, top[0]
    print("OK: build, incremental indexing, pagination and search behave as expected")


if __name__ == "__main__":
    main()
