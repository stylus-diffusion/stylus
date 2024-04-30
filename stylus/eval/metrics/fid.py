import argparse
from cleanfid import fid as cfid
from pathlib import Path
import os
from datetime import datetime
from PIL import Image, ImageFile
from concurrent.futures import ProcessPoolExecutor as Pool
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from tqdm import tqdm
import pandas as pd
import torch

from torchvision.transforms import functional as F
from torchmetrics.image.fid import FrechetInceptionDistance
                                                                                                                                                                                                             
ImageFile.LOAD_TRUNCATED_IMAGES = True 

def update_fid(dataset ,reference_path: str, batch_size = 256, n_proc = 6, clean=True, clip_based=False):
    """
    Helper method to process a batch of data.

    dataset - pd.Dataframe, Dataset of fake images
    reference_path - str, Path to the reference real images.
    """
    if clean:
        fake_images = [path for path in dataset['fake_image_path'].values]
        real_images = [f'{reference_path}/{path}' for path in dataset['real_image_path'].values]
        assert (len(fake_images) == len(real_images)), "Can't do fid with different number of images"
        if clip_based:
            name="fid_clip"
            model = "clip_vid_b_32"

        else:
            name = "fid"
            model = "inception_v3"

        fid_score = cfid.compute_fid(fake_images, real_images, batch_size = batch_size, num_workers = n_proc,mode="clean", model_name=model)

        cfg = dataset['cfg'].values[0]
        alg = dataset['alg'].values[0]
    else:

        fid = FrechetInceptionDistance(normalize=True, compute_with_cache=True).cuda()
        with torch.no_grad(): 
            for i in tqdm(range(0, len(dataset), batch_size)):
                batch = dataset[i:min(i + batch_size, len(dataset))] 
    
                # Load in images from the dataset as RGB images (standard procedure)
                fake_images = [Image.open(path).convert("RGB") for path in batch['fake_image_path'].values]
                real_images = [Image.open(f'{reference_path}/{path}').convert("RGB") for path in batch['real_image_path'].values]
                
                fake_images = [image.resize((512,512)) for image in fake_images]
                
                fake_images = [torch.from_numpy(np.array(i)).unsqueeze(0).permute(0,3,1,2) for i in fake_images]
                real_images = [torch.from_numpy(np.array(i)).unsqueeze(0).permute(0,3,1,2) for i in real_images]
            
                # Pull our images from the dataset
                fake_images = torch.cat([F.center_crop(image, (512,512)).cuda() for image in fake_images])
                real_images = torch.cat([F.center_crop(image, (512,512)).cuda() for image in real_images])
                
                # Update fake images
                fid.update(fake_images, real=False)
                # Update real images
                fid.update(real_images, real=True)
                # Minimize GPU memory.
                torch.cuda.empty_cache()
            fid_score = fid.compute()
        # Get cfg value and alg from one row
        cfg = batch['cfg'].values[0]
        alg = batch['alg'].values[0]
        fid_score = fid_score.detach().cpu().numpy()

        name = "fid"
    processed_df = pd.DataFrame([{
    'cfg': cfg,
    'alg': alg,
    name: fid_score,
    }])
    print(processed_df)
    return processed_df

def fid_score_fn(args): 
    """
    eval_path --> path of AI generated data we would like to evaluate
    """
    images_df, reference_path, gpu_id, num_processes, clip_based = args
    torch.cuda.set_device(gpu_id)
    start_time = datetime.now()
    reference_path = Path(reference_path)
    # Assuming coco the images are all weird dims so crop only 
    final_df = update_fid(images_df, reference_path, batch_size=256, n_proc=num_processes, clip_based=clip_based)
    done_scoring = datetime.now()
    print('Time Scoring (hh:mm:ss.ms) {}'.format(done_scoring - start_time))
    return final_df

def compute_fid_score(dataset_path, reference_path, threshold=None, num_processes = 4, clip_based=False, single_process = True):
    dataset_path = Path(dataset_path)
    reference_path = Path(reference_path)
    if threshold is not None:
        images_df = pd.read_csv(f'{dataset_path}/images_filter_{threshold}.csv')
    else:
        images_df = pd.read_csv(f'{dataset_path}/images_filter.csv')
    # Group by alg, cfg
    img_groups = [groups for _, groups in images_df.groupby(['cfg', 'alg'])]
    args = [(chunk, reference_path, idx % torch.cuda.device_count(), num_processes, clip_based) for idx, chunk  in enumerate(img_groups)]

    no_concurrency=False
    single_process = False
    
    if no_concurrency:
        results  = [fid_score_fn(arg) for arg in args]
    elif single_process:
        results = []
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(fid_score_fn, arg) for arg in args] 
            with tqdm.tqdm(total=len(args)) as pbar:
                for future in as_completed(futures):
                    try:
                        results.append( future.result())
                    except Exception as exc:
                        print(f'Generated an exception: {exc}')
                    pbar.update(1)
    else:
        with Pool(max_workers=num_processes) as pool:
            results = pool.map(fid_score_fn, args)
        results = list(results)
    consolidated_df = pd.concat(results, ignore_index=True)
    if threshold is not None:
        pareto_csv = f'{dataset_path}/pareto_{threshold}.csv'
    else:
        pareto_csv = f'{dataset_path}/pareto.csv'
    # Save the final DataFrame (merge)
    if os.path.exists(pareto_csv):
        pareto_df = pd.read_csv(pareto_csv)
        pareto_df = pd.merge(pareto_df, consolidated_df, on=['cfg', 'alg'])
    else:
        pareto_df = consolidated_df
    pareto_df.to_csv(pareto_csv, index=False)

if __name__ == "__main__": 
    # python metrics/fid.py --dataset_path ~/coco_realistic_output  --reference_path ~/coco/coco_rank
    parser = argparse.ArgumentParser(description='Process some arguments for Lora-Rank.')
    parser.add_argument('--dataset_path', required=True, help='Path to the dataset')
    parser.add_argument('--reference_path', required=True, help='Path to the reference')
    parser.add_argument('--thres', '-t', type=str, default=None, help='Threshold for filtering')
    # reference_path ~/coco/coco_rank/
    # dataset_path ~/coco_realistic_output/
    args = parser.parse_args()
    dataset_path = Path(args.dataset_path)
    reference_path = Path(args.reference_path)

    # Compute FID score
    compute_fid_score(dataset_path, reference_path, threshold=args.thres)
