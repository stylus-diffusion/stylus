# Experiments

We cover the steps to run and replicate our experiments.

## Setup

Download COCO datset and preprocess it (samples 10K images).
```
cd stylus/eval/preprocess
# Downloads COCO in ~/coco.
./download.sh

# Subsample 10K images from COCO and creates a generated captions CSV file (to plug into Stylus `batch_prompts.py`)
# Note that if captions_path already exists, it will sample images in output directory from that captions path.
python subsample_coco.py --outdir ~/coco/coco_rank --reference_img ~/coco/val2014 --captions_path ~/coco_rank.csv
```

The Parti prompts dataset follows similar steps.

NOTE: Our experiment configurations are stored in `configs`. We assume running 16 replicas for Stable Diffusion Web UI (2 per GPU, 8xA100-80GB in our experiments, `./setup/launch_sd.sh 16`). If a different # of replicas are used, modify the config to reflect the # of replicas.
The default config is located in `configs/default_config.yaml`, with detailed descriptions of each field.

NOTE: Some of these experiments can take days, if not weeks, to complete (espeically Pareto curve experiments), as SD Web UI does not support batching across different prompts.

## CLIP/FID Pareto Curve Experiments

```
# Output generated images in `~/coco_output`. Replace `prompt_file` in the config with your generated CSV.
python batch_prompts.py --config configs/coco_pareto.yaml
```

To calculate CLIP and FID scores, run:
```
cd stylus/eval
# Replace --captions_path with your generated captions CSV from Setup section.
# This script calls `metrics/clip.py` and `metrics/fid.py`.
python pareto.py --dataset_path ~/coco_output --captions_path ~/stylus/datasets/coco_rank.csv --reference_path ~/coco/coco_rank
```

Our Pareto scripts assume 8 GPUs. Any less, the program will OOM. If the user has insufficient GPUs, reduce the number of processes used in `metrics/clip.py` and `metrics/fid.py`.

The pareto scripts stores the CLIP and FID scores in `~/coco_output/pareto.csv`. Use this CSV path to plot the figure in `plots/pareto.ipynb`.

## Human Eval

Our human eval evaluates Stylus over different checkpoints (Realistic-Vision/Counterfeit) and datasets (COCO/PartiPrompts). Our config files are in `configs/human_eval`. Run:
```
# COCO, Realistic-Vision
python batch_prompts.py --config configs/human_eval/coco_realistic.yaml
# COCO, Counterfeit
python batch_prompts.py --config configs/human_eval/coco_counterfeit.yaml
# Parti Prompt, Realistic-Vision
python batch_prompts.py --config configs/human_eval/parti_realistic.yaml
# Parti Prompt, Counterfeit
python batch_prompts.py --config configs/human_eval/parti_counterfeit.yaml
```
Then, to create an Excel sheet for humans to evaluate their preferences, modify and run `stylus/eval/excel_human_eval.py`.

Finally, plotting scripts are in `plots/human_eval.ipynb`.

## GPT 4V: VLM as a Judge

We use GPT4V to judge the result of the image generation. We judge based on three different metrics: diversity, quality, and alignment. 

```
python gpt4v/gpt_4v.py --task <diversity|quality|alignment> --input_path ~/parti_output
```

To tabulate the results, 
```
python gpt4v/tabulate.py --task <diversity|quality|alignment>
```

## Diversity Experiments

Our experiments evaluate diversity (dFID) over Parti Prompts dataset and the Realistic Vision dataset. Run:
```
python batch_prompts.py --config configs/diversity/image_diversity.yaml.
```

The diversity plots for dFID win rate and dFID over prompt lengths are in `plots/gpt4_eval.ipynb` and `plots/diversity.ipynb`.

## Retrieval Ablation

Retriever-only (RAG), Reranking and Random baslines are stored in `configs/ablate_retrieval`. Run:
```
# Retriever-only (RAG)
python batch_prompts.py --config configs/ablate_retrieval/coco_rag.yaml
# Rerankers, requires COHERE_API_KEY
python batch_prompts.py --config configs/ablate_retrieval/coco_rerank.yaml
# Random
python batch_prompts.py --config configs/ablate_retrieval/coco_random.yaml
```

Similar to the Pareto curve experiments, the CLIP and FID experiments can be computed by running:
```
cd stylus/eval
# Store CLIP/FID results in `[OUTPUT_IMAGE_DIR]/pareto.csv`.
python pareto.py --dataset_path [OUTPUT_IMAGE_DIR] --captions_path ~/stylus/datasets/coco_rank.csv --reference_path ~/coco/coco_rank
```

## Stylus Inference Times

See `plots/time.ipynb` to see how we plot Stylus over different stages of its pipeline.









