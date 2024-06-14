# This python file samples COCO images from the VAL2014 dataset and saves it to a CSV file and moves the sampled images into
# a specified directory.
# Example Command:
# python subsample_coco.py --captions_path=~/stylus/datasets/coco_rank.csv --outdir ~/coco/coco_rank --reference_img ~/coco/val2014 --downsample 10000
import argparse
from pathlib import Path
import shutil
import json
import pandas as pd
import os

COCO_DIR = os.path.expanduser('~/coco')
CAPTIONS_COCO = f'{COCO_DIR}/annotations/captions_val2014.json'

def _get_absolute_path(path):
    path =  os.path.expanduser(path)
    # Also unfold relativ epaths
    path = os.path.abspath(path)
    return path

def main (args):
    """
    Sample COCO images based on captions file or specified captions.
    """
    args.captions_path = _get_absolute_path(args.captions_path)
    args.outdir = _get_absolute_path(args.outdir)
    args.reference_img = _get_absolute_path(args.reference_img)
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir, exist_ok=True)
    
    if not os.path.exists(args.captions_path): 
        # Creete the Captions CSV file.
        data = json.load(open(CAPTIONS_COCO))
        images = data['images']
        annotations = data['annotations']
        df = pd.DataFrame(images)
        df_annotations = pd.DataFrame(annotations)
        df = df.merge(pd.DataFrame(annotations), how='left', left_on='id', right_on='image_id')

        # keep only the relevant columns
        df = df[['file_name', 'caption']]
        # shuffle the dataset
        df = df.sample(frac=1)
        # remove duplicate images
        df = df.drop_duplicates(subset='file_name')
        n_samples = args.downsample
        df_sample = df.sample(n_samples)
        # save df_sample
        df_sample.to_csv(args.captions_path, index=False)
    all_captions = pd.read_csv(open(Path(args.captions_path), "r")) 
    all_captions = all_captions.loc[:, ~all_captions.columns.str.contains('^Unnamed')]
    all_captions.to_csv(args.captions_path, index=False)  
    downsample = all_captions[:args.downsample]
    for i,row in downsample.iterrows():
        caption  = row['caption']
        path = row['file_name']
        assert (Path(args.reference_img) / path).exists()
        shutil.copy(Path(args.reference_img) /  path ,Path(args.outdir) / path )
    
    
if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description='Process some arguments for Lora-Rank.')
    parser.add_argument('--outdir', default='./out', help='Path to the output CSV file')
    parser.add_argument('--captions_path', required=True, default=None, help='Path to the captions')
    parser.add_argument('--reference_img', required=True, help='Path to the reference')
    parser.add_argument('--downsample', type=int, default=10000, help='Downsample rate')
    args = parser.parse_args()
    main(args)
