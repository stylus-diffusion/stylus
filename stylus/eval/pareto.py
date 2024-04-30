from pathlib import Path
import argparse

from stylus.eval.metrics.clip import compute_clip_score
from stylus.eval.metrics.fid import compute_fid_score

import torch

# Takes around 10-15 minutes to run on a big machine.
def main(args):
    dataset_path = args.dataset_path
    captions_path = args.captions_path
    reference_path = args.reference_path
    run_clip = False
    if run_clip:
       print ("Computing CLIP...")
       compute_clip_score(Path(dataset_path),captions_path)
       torch.cuda.empty_cache()
    print ("Computing FID...")
    compute_fid_score(dataset_path, reference_path= reference_path, clip_based=False, threshold = args.thres)

    
if __name__ == "__main__": 
    # python pareto.py --dataset_path ~/coco_output  --captions_path ~/stylus/datasets/coco_rank.csv --reference_path ~/coco/coco_rank
    parser = argparse.ArgumentParser(description='Process some arguments for Lora-Rank.')
    parser.add_argument('--dataset_path', required=True, help='Path to the generated dataset')
    parser.add_argument('--captions_path', required=True, help='Path to the captions (i.e. datasets/coco_rank.csv)')
    parser.add_argument('--reference_path', required=True, help='Path to the reference')
    parser.add_argument('--thres', '-t', type=str, default=None, help='Threshold for filtering')

    args = parser.parse_args()
    main(args)
