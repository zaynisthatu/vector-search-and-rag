"""Semantic search over the collection.

    python search.py "girl dancing"
"""
import sys
import common


def search(client, model, query, limit=5, collection=None):
    vec = model.encode([query], normalize_embeddings=True)[0].tolist()
    result = client.query_points(collection or common.COLLECTION, query=vec, limit=limit, with_payload=True)
    return [(pt.score, pt.payload) for pt in result.points]


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "car"
    for score, payload in search(common.get_client(), common.get_model(), q):
        print(f"{score:.3f}  {payload['username']}: {payload['caption'][:80]}")
