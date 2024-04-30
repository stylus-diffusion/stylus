import concurrent.futures
import importlib
import os
import pickle
import sys
import threading
from dataclasses import dataclass
from typing import List, Optional

import requests
from bs4 import BeautifulSoup
from stylus.refiner.fetch_catalog import fetch_civit_model_catalog
import tqdm

FILE_LOCK = threading.Lock()
ADAPTERS_FILE = 'cache/sd_adapters.pkl'

module_modifications = {
    'autoadapter':
    'stylus',
    'autoadapter.precompute':
    'stylus.refiner',
    'autoadapter.precompute.process_adapters':
    'stylus.refiner.fetch_adapter_metadata',
}
for module, new_module in module_modifications.items():
    sys.modules[module] = importlib.import_module(new_module)


@dataclass
class AdapterInfo:
    """Dataclass for storing adapter metadata.
    
    Attributes:
        adapter_id (str): The unique identifier of this adapter.
        alias (str): The alias or alternative name used for the adapter.
        title (str): The title given to this adapter.
        base_model (str): The adapter's base model (e.g. SD v1.5).
        description (str): Text description of the adapter.
        tags (List[str]): A list of tags associated with the adapter.
        trigger_words (List[str]): A list of trigger words.
        stats (dict): A dictionary containing statistical data about the adapter.
        download_url (str): URL from where the adapter can be downloaded.
        llm_description (str, optional): Refiner description of the adapter. (see refiner_vllm.py)
        image_urls (List[str], optional): URLs of example images related to the adapter.
        image_prompts (List[str], optional): Prompts for each image in `image_urls`.
        image_negative_prompts (List[str], optional): Negative prompts for each image in `image_urls`.
        weight (float, optional): Adatper's weight, default is 0.8.
    """
    adapter_id: str
    alias: str
    title: str
    base_model: str
    description: str
    tags: List[str]
    trigger_words: List[str]
    stats: dict
    download_url: str
    llm_description: Optional[str] = None
    image_urls: Optional[List[str]] = None
    image_prompts: Optional[List[str]] = None
    image_negative_prompts: Optional[List[str]] = None
    weight: float = 0.8

    def __repr__(self):
        return (
            f"AdapterInfo(adapter_id={self.adapter_id!r}, alias={self.alias!r}, "
            f"title={self.title!r}, description={self.description!r}, "
            f"llm_description={self.llm_description!r})")


# NOTE: Civit Rest AI for images relies on Postgres DB. This may be overloaded or even broken if run during peak hours.
# Try running this when there is less traffic. Sometimes, the Rest API is broken.
def fetch_adapter_image_metadata(adapter_id: str):
    """Fetches image metadata for an adapter from the Civit AI REST API."""
    try:
        response = requests.get(
            f'https://civitai.com/api/v1/images?modelId={adapter_id}&limit=100'
        )
        response.raise_for_status()
        items = response.json().get('items', [])
        adapter_image_urls = [
            item['url'] for item in items
            if item.get('meta') and isinstance(item['meta'], dict)
        ]
        adapter_image_prompts = [
            item['meta'].get('prompt', 'None') for item in items
            if item.get('meta') and isinstance(item['meta'], dict)
        ]
        adapter_image_negative_prompts = [
            item['meta'].get('negativePrompt', 'None') for item in items
            if item.get('meta') and isinstance(item['meta'], dict)
        ]
        return adapter_image_urls, adapter_image_prompts, adapter_image_negative_prompts
    except requests.RequestException as e:
        return None, None, None


def parse_catalog_entry(model_data: dict,
                        adapter_type: str = 'LORA',
                        base_model: str = 'SD 1.5') -> Optional[AdapterInfo]:
    """Parses catalog data into an AdapterInfo object."""
    adapter_id = model_data.get('id')
    adapter_title = model_data.get('name')
    adapter_description = BeautifulSoup(model_data.get('description',
                                                       '')).get_text()
    adapter_tags = model_data.get('tags', [])
    adapter_model_versions = model_data.get('modelVersions', [])
    adapter_stats = model_data.get('stats', {})

    if adapter_type != model_data.get('type') or model_data.get('nsfw', False):
        return None
    for version in adapter_model_versions:
        if version.get('baseModel') == base_model:
            for file_info in version.get('files', []):
                if file_info['name'].endswith('.safetensors'):
                    adapter_alias = os.path.splitext(file_info['name'])[0]
                    adapter_trigger_words = version.get('trainedWords', [])
                    adapter_download_url = file_info.get('downloadUrl')
                    adapter_image_urls, adapter_image_prompts, adapter_image_negative_prompts = fetch_adapter_image_metadata(
                        adapter_id)
                    if adapter_image_urls is not None:
                        return AdapterInfo(adapter_id=adapter_id,
                                           alias=adapter_alias,
                                           title=adapter_title,
                                           base_model=base_model,
                                           description=adapter_description,
                                           tags=adapter_tags,
                                           trigger_words=adapter_trigger_words,
                                           stats=adapter_stats,
                                           download_url=adapter_download_url,
                                           image_urls=adapter_image_urls,
                                           image_prompts=adapter_image_prompts,
                                           image_negative_prompts=
                                           adapter_image_negative_prompts)
    return None


def fetch_adapter_metadata(base_model: str = 'SD 1.5',
                           adapter_type: str = 'LORA',
                           save: bool = True,
                           skip_load: bool = False) -> List[AdapterInfo]:
    """Fetches adapter metadata from the Civit catalog."""
    cur_file_path = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(os.path.dirname(cur_file_path), ADAPTERS_FILE)

    print('Fetching adapter metadata...')
    if os.path.exists(file_path) and not skip_load:
        with FILE_LOCK:
            with open(file_path, 'rb') as file:
                return pickle.load(file)

    catalog = fetch_civit_model_catalog()
    adapters = []
    with tqdm.tqdm(total=len(catalog)) as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=64) as executor:
            futures = [
                executor.submit(parse_catalog_entry, model_data, adapter_type,
                                base_model) for model_data in catalog
            ]
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if result:
                    adapters.append(result)
                pbar.update(1)

    if save:
        with FILE_LOCK:
            with open(file_path, 'wb') as file:
                pickle.dump(adapters, file)

    return adapters


if __name__ == '__main__':
    adapters = fetch_adapter_metadata()
