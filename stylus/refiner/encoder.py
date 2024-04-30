import numpy as np
from functools import partial
import os
import time
from typing import List, Type

from sentence_transformers import SentenceTransformer
import tiktoken
import torch
from transformers import CLIPProcessor, CLIPModel
import openai

from stylus.refiner.fetch_adapter_metadata import AdapterInfo, fetch_adapter_metadata

EMBEDDING_CTX_LENGTH = 8191
EMBEDDING_ENCODING = 'cl100k_base'
EMBEDDING_CACHE_PATH = 'cache/sd_embeddings.npy'


def truncate_text_tokens(text,
                         encoding_name=EMBEDDING_ENCODING,
                         max_tokens=EMBEDDING_CTX_LENGTH) -> str:
    """Truncate a string to have `max_tokens` according to the given encoding."""
    encoding = tiktoken.get_encoding(encoding_name)
    return encoding.encode(text)[:max_tokens]


class Embedding:

    def __init__(self, model_id: str):
        self.model_id = model_id

    def __call__(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        raise NotImplementedError("Subclasses must implement this method.")


# NOTE: CLIP embeddings generally do not work.
class ClipEmbedding(Embedding):

    def __init__(self, model_id="openai/clip-vit-base-patch32"):
        super().__init__(model_id)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model = CLIPModel.from_pretrained(model_id).to(device)
        self.processor = CLIPProcessor.from_pretrained(model_id)

    def __call__(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        self.model.eval()  # Set the model to evaluation mode
        embeddings = []  # This will store the embeddings of all texts
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            inputs = self.processor(text=batch_texts,
                                    return_tensors="pt",
                                    padding=True,
                                    truncation=False).to(self.device)
            with torch.no_grad():
                batch_embeddings = self.model.get_text_features(**inputs)
                embeddings.append(
                    batch_embeddings.cpu().numpy())  # Convert to CPU and numpy
        return embeddings


class SentenceTransformerEmbedding(Embedding):

    def __init__(self, model_id="all-MiniLM-L6-v2"):
        super().__init__(model_id)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model = SentenceTransformer(model_id).to(self.device)

    def __call__(self, texts: List[str], batch_size: int = 32):
        self.model.eval()  # Set the model to evaluation mode
        embeddings = []  # This will store the embeddings of all texts
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

            with torch.no_grad():
                batch_embeddings = self.model.encode(batch_texts)
                embeddings.append(batch_embeddings)  # Convert to CPU and numpy
            torch.cuda.empty_cache()
        # Concatenate all batch embeddings into a single numpy array
        embeddings = np.concatenate(embeddings, axis=0)
        return embeddings


class OpenAIEmbedding(Embedding):

    def __init__(self, model_id="text-embedding-3-large"):
        super().__init__(model_id)
        self.client = openai.OpenAI()

    def __call__(self, texts: List[str], batch_size: int = 128):
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            for j, text in enumerate(batch_texts):
                batch_texts[j] = truncate_text_tokens(text)
            while True:
                try:
                    batch_embeddings = self.client.embeddings.create(
                        input=batch_texts, model=self.model_id).data
                    break
                except openai.RateLimitError:
                    time.sleep(1)
                    continue
            for embedding in batch_embeddings:
                embeddings.append(embedding.embedding)
        # Concatenate all batch embeddings into a single numpy array
        embeddings = np.array(embeddings)
        return embeddings


def get_embedding_cls(embedding_type: str) -> Type[Embedding]:
    """
    Get the embedding class for the specified embedding type.
    Returns a default OpenAI embedding if the specified type is not recognized.
    """
    embedding_dict = {
        "clip":
        ClipEmbedding,
        "sentence_transformer":
        SentenceTransformerEmbedding,
        "salesforce":
        partial(SentenceTransformerEmbedding,
                model_id="Salesforce/SFR-Embedding-Mistral"),
        "openai":
        OpenAIEmbedding,
    }
    return embedding_dict.get(embedding_type, OpenAIEmbedding)


def compute_embeddings(texts: List[str],
                       embedding_cls: Type[Embedding] = OpenAIEmbedding,
                       batch_size: int = 64) -> np.ndarray:
    """
    Compute embeddings for the provided texts using the specified embedding class.
    """
    return embedding_cls()(texts, batch_size=batch_size)


def fetch_adapter_embeddings(adapter_list: List[AdapterInfo],
                             embedding_type: str = 'openai',
                             prefetch: bool = True):
    """
    Fetch embeddings for a list of adapters.
    """
    cur_file_path = os.path.abspath(__file__)
    cur_dir = os.path.dirname(cur_file_path)
    file_path = os.path.join(os.path.dirname(cur_dir), EMBEDDING_CACHE_PATH)

    if os.path.exists(file_path) and prefetch:
        return np.load(file_path, allow_pickle=True)

    embedding_cls = get_embedding_cls(embedding_type)
    # Fetch text descriptions from all LORAs.
    tag_str_list = [",".join(p.tags) for p in adapter_list]
    trigger_str_list = [",".join(p.trigger_words) for p in adapter_list]
    texts = []
    for idx, adapter in enumerate(adapter_list):
        # Use VLM description (see vlm.py), otherwise, use base model card description.
        description = adapter.llm_description if adapter.llm_description else adapter.description
        text = (
            f"Convert Stable Diffusion finetuned adapter description into an embedding for search: "
            f"Title: {adapter.title}; Description: {description}; Tags: {tag_str_list[idx]};"
        )
        texts.append(text)

    batch_size = 2 if embedding_type == 'salesforce' else 128
    embeddings = compute_embeddings(texts,
                                    embedding_cls,
                                    batch_size=batch_size)

    if prefetch:
        np.save(file_path, embeddings)
    return embeddings


if __name__ == '__main__':
    adapter_list = fetch_adapter_metadata(
        base_model="SD 1.5",
        adapter_type="LORA",
    )
    fetch_adapter_embeddings(adapter_list, prefetch=True)
