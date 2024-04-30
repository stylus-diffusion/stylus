import pandas as pd
from pathlib import Path
import os

from stylus.utils import prompt_to_filename

def find_image_paths(root_dir):
    # List of file extensions for images you want to check for
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.tiff', '.bmp', '.svg']
    image_paths = []

    # Walk through all directories and files in root_dir
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            # Check if the file extension is one of the image types
            if any(filename.lower().endswith(ext) for ext in image_extensions):
                # Construct absolute path and add it to the list
                image_path = os.path.abspath(os.path.join(dirpath, filename))
                image_paths.append(image_path)
    return image_paths

def extract_image_id(image_path: str):
    cfg_path, prompt_str, alg_type, attempt_str = image_path.split(os.sep)[-4:]
    cfg_value = cfg_path.split("_")[-1]
    attempt_value = attempt_str.split("_")[0]
    return float(cfg_value), prompt_str, alg_type, int(attempt_value)


def fetch_or_generate_image_df(dataset_path: str, captions_path: str):
    image_df_path = f'{dataset_path}/images.csv'
    if os.path.exists(image_df_path):
        return pd.read_csv(image_df_path)
    
    if captions_path is None:
        raise ValueError("Captions path is required to generate image dataframe")

    dataset_path = Path(dataset_path)
    captions = pd.read_csv(captions_path)
    captions['file_prompt'] = captions.apply(prompt_to_filename, axis=1)
    all_captions = captions['file_prompt']
    # if clip and fid not in dictionary, add
    image_paths = find_image_paths(dataset_path)
    image_values = [extract_image_id(image_path) for image_path in image_paths]
    # Create new output csv and save (image_path, cfg_value, prompt_str, attempt_value, clip_score, fid_score)
    output_list = []
    for image_path, image_value in zip(image_paths, image_values):
        cfg_value, prompt_str, alg_type, attempt_value = image_value
        prompt = None
        if prompt_str in all_captions.values:
            # Fetch the actual prompt
            prompt = captions[captions['file_prompt'] == prompt_str]['caption'].values[0]
        if prompt is None:
            raise ValueError(f"Prompt not found for {image_path}")
        # Get real image path from original captions
        real_image_row = captions[captions['caption'] == prompt]
        real_image_row = real_image_row.iloc[0]
        real_image_path = real_image_row['file_name']
        ranked_adapters = real_image_row['ranked_adapters']
        output_list.append({'real_image_path': real_image_path, 'fake_image_path': image_path, 'ranked_adapters': ranked_adapters, 'cfg': cfg_value, 'prompt': prompt, 'alg': alg_type, 'id': attempt_value, 'clip': None,})
    output_df = pd.DataFrame(output_list)
    output_df.to_csv(f'{dataset_path}/images.csv', index=False)
    return output_df
