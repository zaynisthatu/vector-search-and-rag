"""Embed only the posts that are not in Qdrant yet (matched on payload.post_id).

    python index_new_posts.py
"""
from qdrant_client.models import PointStruct
import common


def existing_state(client, collection, page=1000):
    """Read every stored point page by page: the set of post_ids and the highest point id."""
    ids, max_id, offset = set(), -1, None
    while True:
        points, offset = client.scroll(collection, limit=page, offset=offset, with_payload=True)
        for pt in points:
            ids.add(pt.payload["post_id"])
            max_id = max(max_id, pt.id)
        if offset is None:
            return ids, max_id


def index_new_posts(client, model, db_path=None, collection=None, batch_size=128):
    collection = collection or common.COLLECTION
    existing_ids, max_id = existing_state(client, collection)
    print(f"Already in Qdrant: {len(existing_ids)}")
    posts = common.load_posts(db_path)
    print(f"Total in DB: {len(posts)}")
    new_posts = [p for p in posts if p["post_id"] not in existing_ids]
    print(f"New to embed: {len(new_posts)}")
    if not new_posts:
        print("Everything is already indexed")
        return 0
    embeddings = model.encode([common.post_text(p) for p in new_posts], batch_size=batch_size,
                              show_progress_bar=False, normalize_embeddings=True)
    start = max_id + 1
    points = [PointStruct(id=start + i, vector=embeddings[i].tolist(), payload=common.payload(p))
              for i, p in enumerate(new_posts)]
    client.upsert(collection_name=collection, points=points)
    print(f"{len(points)} new vectors added")
    return len(points)


if __name__ == "__main__":
    index_new_posts(common.get_client(), common.get_model())
