import os
import json
import requests
import tqdm

CIVIT_AI_KEY = os.environ.get('CIVIT_API_KEY')
CIVIT_AI_URL = 'https://civitai.com/api/v1/models'
CATALOG_FILE_PATH = 'cache/civit_catalog.json'
CIVIT_AI_FILTERS = {f'token={CIVIT_AI_KEY}'}


def fetch_civit_model_catalog(file_path: str = CATALOG_FILE_PATH) -> list:
    """Fetches/downloads the Civit AI Model catalog.

    Args:
        file_path: The file path where the catalog will be stored.

    Returns:
        A list of dicts representing the model catalog.
    """
    cur_file_path = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(os.path.dirname(cur_file_path), file_path)

    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            return json.load(file)

    print('Downloading Civit AI Model Catalog...')
    request_url = f'{CIVIT_AI_URL}?{"&".join(CIVIT_AI_FILTERS)}'
    json_data = []

    # Impossible to determine how many total pages there are
    # so we'll just set the total to 1500.
    with tqdm.tqdm(total=1500) as progress_bar:
        while True:
            response = requests.get(request_url).json()
            response_metadata = response.get('metadata', {})
            if "items" not in response or not response_metadata:
                break

            json_data.extend(response['items'])
            progress_bar.update(1)
            if 'nextPage' not in response_metadata:
                break
            request_url = response_metadata['nextPage']

    with open(file_path, 'w') as file:
        json.dump(json_data, file)

    print('Finished downloading Civit AI Model Catalog...')
    return json_data


if __name__ == '__main__':
    adapters = fetch_civit_model_catalog()
