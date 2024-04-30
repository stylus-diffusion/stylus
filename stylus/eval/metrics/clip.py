import argparse
from pathlib import Path
import torch
import os
import pandas as pd
from PIL import Image
import numpy as np
from torchmetrics.functional.multimodal.clip_score import _clip_score_update, _get_clip_model_and_processor
from tqdm import tqdm
import os
from multiprocessing import Pool

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


CLIP_MODEL_NAME = "openai/clip-vit-large-patch14"
seed = 0
generator = torch.manual_seed(seed)

from stylus.eval.utils import fetch_or_generate_image_df

def _aggregate_clip_score(dataset_path, image_df):
    # Take the max clip score from groups of [cfg, prompt, and alg_type].
    # Filter only for rows with max clip score
    # Return the filtered dataframe
    # Keep all columns
    filter_df = image_df.loc[image_df.groupby(['cfg', 'prompt', 'alg'])['clip'].idxmax()]
    filter_df.to_csv(f'{dataset_path}/images_filter.csv', index=False)

    # Save top 2

    # Also create a new dataframe that group by alg, cfg (gets the average clip score)
    avg_df = filter_df.groupby(['cfg', 'alg'])['clip'].mean().reset_index()
    avg_df.to_csv(f'{dataset_path}/pareto.csv', index=False)

    return filter_df


def clip_score_fn(args):
    chunk, batch_size, gpu_id = args
    torch.cuda.set_device(gpu_id)  # Set the GPU for the current process
    
    # Initialize model and processor for this worker
    model, processor = _get_clip_model_and_processor(CLIP_MODEL_NAME)
    model = model.cuda()
    model.eval()
    
    # Process the chunk
    processed_chunk = []  # Placeholder for storing results
    with torch.no_grad():
        for i in tqdm(range(0, len(chunk), batch_size)):
            batch_df = chunk.iloc[i:i+batch_size]
            batch_images = [np.array(Image.open(path).convert("RGB")) for path in batch_df['fake_image_path'].values]
            batch_images = np.stack(batch_images)
            batch_images = torch.from_numpy(batch_images).cuda().permute(0, 3, 1, 2)
            batch_prompts = batch_df['prompt'].values.tolist()

            score, _ = _clip_score_update(batch_images, batch_prompts, model, processor)

            # Update the chunk with the new scores
            for s, idx in zip(score.detach().cpu().numpy(), batch_df.index):
                processed_chunk.append((idx, float(s)))  # Store the index and the score
            
            # Clear torch cache
            torch.cuda.empty_cache()

    # Convert the processed results back into a DataFrame structure
    processed_df = pd.DataFrame(processed_chunk, columns=['index', 'clip'])
    return processed_df

# Compute clip scores in parallel
def compute_clip_score(dataset_path, captions_path, batch_size=256, num_processes=16):
    dataset_path = Path(dataset_path)
    image_df = fetch_or_generate_image_df(dataset_path, captions_path)

    # Split the dataframe into chunks
    chunks = np.array_split(image_df, num_processes)
    # Prepare arguments for each process: each chunk, dataset_path, captions_path, and GPU ID
    args = [(chunks[i], batch_size, i % torch.cuda.device_count()) for i in range(num_processes)]

    with Pool(processes=num_processes) as pool:
        results = pool.map(clip_score_fn, args)
    
    # Consolidate the results into a single DataFrame
    consolidated_df = pd.concat(results, ignore_index=True)

    # Update the original DataFrame with the scores
    for idx, row in consolidated_df.iterrows():
        image_df.loc[row['index'], 'clip'] = row['clip']

    # Save the final DataFrame
    image_df.to_csv(f'{dataset_path}/images.csv', index=False)
    return _aggregate_clip_score(dataset_path, image_df)

if __name__ == "__main__": 
    # python clip.py --dataset_path ~/coco_realistic_output  --captions_path ~/stylus/datasets/coco_rank.csv
    parser = argparse.ArgumentParser(description='Process some arguments for Lora-Rank.')
    parser.add_argument('--dataset_path', required=True, help='Path to the dataset.')
    parser.add_argument('--captions_path', required=True, help='Path to the CSV file of subsampled captions.')
    args = parser.parse_args()
    dataset_path = args.dataset_path
    captions_path = args.captions_path
    # Just testing clip score lol.
    compute_clip_score(Path(dataset_path), Path(captions_path)) 
