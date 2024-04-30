import os
from pinecone import Pinecone


def query(vector, top_k=3):
    pc = Pinecone(api_key=os.getenv('PINECONE_KEY'))
    index = pc.Index("stylus-docs")

    query_results = index.query(namespace="ns1",
                                vector=vector,
                                top_k=top_k,
                                include_values=True)

    matches = query_results['matches']
    matches = sorted(matches, key=lambda x: x.score, reverse=True)
    ids = [int(match.id) for match in matches]
    return ids


if __name__ == "__main__":

    query_results1 = query([1.0, 1.5] * 4)
    print(query_results1)
