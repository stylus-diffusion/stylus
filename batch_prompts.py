"""
Script to launch of batch run of prompts with Stylus.
"""
from copy import deepcopy
import argparse
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import tqdm

from stylus.retriever.rag import get_all_adapters
from stylus.launch_stylus import run_stylus
from stylus.utils import blacklist_adapters

from stylus.utils.config import load_config


def batch_run_stylus(config: str):
    """Batch run prompts against Stylus."""
    args = load_config(config)

    prompt_db = pd.read_csv(args.prompt_file)
    prompts = prompt_db["caption"]
    if args.randomize != -1:
        prompts = prompts.sample(n=args.randomize)

    all_adapters = get_all_adapters()

    # Load the rank cache if available.
    if "ranked_adapters" in prompt_db.columns and args.rank_cache:
        filtered_adapters = blacklist_adapters(
            all_adapters, not args.rerank.block_characters)
        all_adapters_lookup = {
            adapter.adapter_id: adapter
            for adapter in filtered_adapters
        }
        caption_to_aid = prompt_db.set_index(
            'caption')['ranked_adapters'].to_dict()
        # Convert selected strings into adapter ids
        rank_cache = {
            caption:
            (eval(selected_loras) if isinstance(selected_loras, str) else None)
            for caption, selected_loras in caption_to_aid.items()
        }

        # Convert adapter ids to adapter objects
        rank_cache = {
            caption: ({
                concept: [
                    all_adapters_lookup[idx] for idx in selected_loras[concept]
                    if idx in all_adapters_lookup
                ]
                for concept in selected_loras.keys()
            } if selected_loras is not None else {})
            for caption, selected_loras in rank_cache.items()
        }
    else:
        rank_cache = None

    # Parallel execution.
    if args.sd_config.parallel:
        # For Pareto curve experiments
        if args.sd_config.grid_cfg:
            cfg_values = args.sd_config.grid_cfg
        else:
            cfg_values = [args.sd_config.cfg]
        cfg_args = [deepcopy(args) for _ in range(len(cfg_values))]
        for idx, cfg in enumerate(cfg_values):
            cfg_args[idx].sd_config.cfg = cfg
        futures = []
        with tqdm.tqdm(total=len(cfg_args) * len(prompts)) as pbar:
            with ThreadPoolExecutor(max_workers=len(args.sd_config.ports)+8) as executor:
                for cfg_arg in cfg_args:
                    for prompt in prompts:
                        future = executor.submit(
                            run_stylus,
                            prompt,
                            cfg_arg,
                            None if not rank_cache else rank_cache[prompt],
                            adapters_cache=all_adapters
                            if not rank_cache else None)
                        futures.append(future)
                for future in as_completed(futures):
                    pbar.update(1)
        return
    # Serial execution.
    for prompt in prompts:
        run_stylus(
            prompt,
            args,
            rank_cache=None if not rank_cache else rank_cache[prompt])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Batch run prompts against Stylus (see datasets/*.csv)')
    parser.add_argument('--config',
                        type=str,
                        help='Stylus config file.',
                        default='configs/default_config.yaml')
    args = parser.parse_args()
    batch_run_stylus(config=args.config)
