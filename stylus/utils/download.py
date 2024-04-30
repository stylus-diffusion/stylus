import requests
import asyncio
import os


def get_lora_cache(lora_cache):
    # Create folder if it doesn't exists
    if not os.path.exists(lora_cache):
        os.makedirs(lora_cache, exist_ok=True)
    return lora_cache


def download_loras(ranked_loras, lora_cache):
    lora_folder = get_lora_cache(lora_cache)
    CIVIT_API_KEY = os.getenv('CIVIT_API_KEY')
    asyncio.run(
        parallel_download(list(ranked_loras.values()), CIVIT_API_KEY,
                          lora_folder))


async def download_weights(lora_url_link, file_name, lora_folder,
                           civit_api_key):
    response = requests.get(f'{lora_url_link}?token={civit_api_key}')
    with open(f'{lora_folder}/{file_name}.safetensors', 'wb') as file:
        file.write(response.content)
    return True


async def parallel_download(all_loras, CIVIT_API_KEY, lora_folder):
    handles = []
    for loras in all_loras:
        for lora in loras:
            lora_url_link = lora.download_url
            file_name = str(lora.adapter_id)
            if os.path.exists(f'{lora_folder}/{file_name}.safetensors'):
                continue
            handles.append(
                download_weights(lora_url_link, file_name, lora_folder,
                                 CIVIT_API_KEY))
    result = await asyncio.gather(*handles)
    assert all(result)
    return all(result)
