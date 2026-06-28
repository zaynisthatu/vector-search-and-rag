"""Full (re)build of the Qdrant collection from the posts table.

Embeds every post with a sentence-transformers model (normalized vectors, cosine distance)
and uploads them to Qdrant. Drops the collection first if it already exists.

    python build_index.py
"""
from qdrant_client.models import Distance, VectorParams, PointStruct
import common


def build_index(client, model, db_path=None, collection=None, batch_size=128):
    collection = collection or common.COLLECTION
    posts = common.load_posts(db_path)
    print(f"Posts: {len(posts)}")
    embeddings = model.encode([common.post_text(p) for p in posts], batch_size=batch_size,
                              show_progress_bar=False, normalize_embeddings=True)
    if client.collection_exists(collection):
        client.delete_collection(collection)
    client.create_collection(collection, vectors_config=VectorParams(size=common.VECTOR_SIZE, distance=Distance.COSINE))
    points = [PointStruct(id=i, vector=embeddings[i].tolist(), payload=common.payload(p)) for i, p in enumerate(posts)]
    client.upsert(collection_name=collection, points=points)
    print(f"{len(points)} vectors uploaded to Qdrant")
    return len(points)


if __name__ == "__main__":
    build_index(common.get_client(), common.get_model())
