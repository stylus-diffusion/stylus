import os
from typing import List
import time
import cohere

from stylus.composer.composer import ADAPTER_CATALOG
from stylus.refiner.fetch_adapter_metadata import AdapterInfo
from stylus.utils import blacklist_adapters

COHERE_API_KEY = os.getenv("COHERE_API_KEY")

def rerank(
    prompt: str,
    top_adapters: List[AdapterInfo],
    rerank_model='rerank-english-v2.0',
    relevance_threshold=0.29,
):
    adapters = blacklist_adapters(top_adapters, enable_characters=False)
    query = f"\"{prompt}\""
    co = cohere.Client(COHERE_API_KEY)
    docs = []
    for idx, adapter in enumerate(adapters):
        if adapter.llm_description:
            adapter_description = adapter.llm_description
        else:
            adapter.description = adapter.description[:1000] if adapter.description else 'None'
        
        adapter_catalog_str = ADAPTER_CATALOG.format(adapter_idx=idx,
                                                        adapter_title=adapter.title,
                                                        adapter_tags=adapter.tags,
                                                        adapter_description=adapter_description)
        docs.append(adapter_catalog_str)

    while True:
        try:
            # Lets get the scores
            results = co.rerank(
                query=query,
                documents=docs,
                top_n=80,
                model=rerank_model,
                return_documents=True
            )  # Change top_n to change the number of results returned. If top_n is not passed, all results will be returned.
            break
        except Exception as e:
            # Rate limit error.
            print(e)
            time.sleep(5)
            continue
    final_adapters = []
    counter_adapters = []
    for idx, r in enumerate(results.results):
        if len(counter_adapters) < 20:
            counter_adapters.append(adapters[r.index])
        if r.relevance_score >= relevance_threshold:
            final_adapters.append(adapters[r.index])
        print(f"Document Rank: {idx + 1}, Document Index: {r.index}")
        print(f"Document: {r.document.text}")
        print(f"Relevance Score: {r.relevance_score:.2f}")
        print("\n")
    # Make sure to return at least 20 adapters.
    if len(final_adapters) < 20:
        final_adapters = counter_adapters
    return final_adapters
