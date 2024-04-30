import pickle
import numpy as np
import os

from pinecone import Pinecone, ServerlessSpec


def upload(vectors):
    pc = Pinecone(api_key=os.getenv('PINECONE_KEY'))
    index = pc.Index("stylus-docs")
    index.upsert(vectors=vectors, namespace="ns1")


def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


if __name__ == "__main__":

    path_npy = "./cache/sd_embeddings.npy"
    embeddings = np.load(open(path_npy, "rb"))
    path_pickle = "./cache/sd_adapters.pkl"
    meta_data = pickle.load(open(path_pickle, "rb"))

    assert len(meta_data) == len(embeddings)

    vectors = [{
        "id": str(meta.adapter_id),
        "values": vec
    } for vec, meta in zip(embeddings, meta_data)]
    for vecs in batch(vectors, 100):

        upload(vecs)
