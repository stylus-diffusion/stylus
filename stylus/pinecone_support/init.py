import os
from pinecone import Pinecone, ServerlessSpec


def create_vdb(dim=3072):
    pc = Pinecone(api_key=os.getenv('PINECONE_KEY'))
    pc.create_index(
        name="stylus-docs",
        dimension=dim,  # Replace with your model dimensions
        metric="cosine",  # Replace with your model metric
        spec=ServerlessSpec(cloud="aws", region="us-east-1"))


if __name__ == "__main__":
    create_vdb()
