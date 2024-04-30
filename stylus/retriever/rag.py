import numpy as np

import torch
import os

from stylus.pinecone_support.query import query
from stylus.refiner.fetch_adapter_metadata import fetch_adapter_metadata
from stylus.refiner.encoder import compute_embeddings, get_embedding_cls, fetch_adapter_embeddings


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.

    :return: Matrix with res[i][j]  = cos_sim(a[i], b[j])
    """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
    return torch.mm(a_norm, b_norm.transpose(0, 1))


def get_all_adapters():
    adapter_list = fetch_adapter_metadata(
        base_model="SD 1.5",
        adapter_type="LORA",
    )
    return adapter_list


def compute_rankings(prompt: str,
                     top_k: int = 100,
                     policy="rank",
                     embedding_type="openai",
                     adapters_cache=None,
                     pinecone=False,
                     debug=False):
    """Computes the top K adapters given a policy and ebmedding class.
    
    This emulates Retrieval Augmented Generation (RAG) for LLMs.
    """
    # Fetch adapters (This will take a while if not cached)
    if adapters_cache is None:
        adapter_list = get_all_adapters()
    else:
        adapter_list = adapters_cache

    if policy == "random":
        top_k_indices = np.random.choice(len(adapter_list),
                                         top_k,
                                         replace=False)
        ranked_adapters = [adapter_list[idx] for idx in top_k_indices]
    elif policy == "rank":
        # Compute prompt embeddings.
        embedding_cls = get_embedding_cls(embedding_type)
        prompt_embedding = compute_embeddings([prompt],
                                              embedding_cls=embedding_cls)
        if pinecone:
            assert os.getenv(
                'PINECONE_KEY'
            ), "PINECONE_KEY is required for pinecone. Add a key to keys.yaml or set it as an environment variable"
            assert embedding_type == "openai", "Only OpenAI embeddings are supported for pinecone"
            top_k_indices = query(prompt_embedding[0].tolist(), top_k=top_k)
        else:
            # Load adapter embeddings
            adapter_embeddings = fetch_adapter_embeddings(
                adapter_list, embedding_type=embedding_type, prefetch=True)
            # Compute similarity between prompt embedding and LORA embeddings.
            cos_sim = cosine_similarity(prompt_embedding,
                                        adapter_embeddings).numpy()
            cos_sim = cos_sim.flatten()
            # Get ranked adapters:
            ranked_adapters = []
            top_k = min(top_k, len(adapter_list))
            top_k_indices = np.argsort(cos_sim)[::-1][:top_k]
        ranked_adapters = [adapter_list[idx] for idx in top_k_indices]
    else:
        raise ValueError(
            f'Policy {policy} not recognized. Please use "random" or "rank"')

    if debug:
        for idx, adapter in enumerate(ranked_adapters):
            print('======================')
            print('Title:', adapter.title)
            print('Cosine Similarity:', cos_sim[top_k_indices[idx]])
            print('Tags:', adapter.tags)
            if adapter.llm_description:
                print('Description:', adapter.llm_description)
            else:
                print('Description:', adapter.description)
            print('Trigger Words:', adapter.trigger_words)

    return ranked_adapters


if __name__ == "__main__":
    no_pinecone = compute_rankings("james bond", top_k=10, debug=True)
    w_pinecone = compute_rankings("james bond, blood", top_k=10, pinecone=True)
    assert no_pinecone == w_pinecone, f"Results should be the same with and without pinecone\n without {no_pinecone} \n with {w_pinecone}"
