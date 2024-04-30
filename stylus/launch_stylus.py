from collections import Counter
import os
import random
import threading
from typing import Dict, List, Optional

from stylus.refiner.fetch_adapter_metadata import AdapterInfo
from stylus.retriever.rag import compute_rankings
from stylus.utils import download_loras, generate_from_prompt, \
    get_masks, sd_api,prompt_builder, prompt_to_filename, Timer

CHECKPOINT_LOCK = threading.Lock()
CHECKPOINT_INITIALIZED = False
SD_WEBUI_LORA_FOLDER = '../stable_diffusion/models/Lora'


def load_checkpoints(ports: List[int], checkpoint_type: str):
    """Loads the model checkpoint on all SD WebUI replicas."""
    with CHECKPOINT_LOCK:
        global CHECKPOINT_INITIALIZED
        # Idempotent, only set once across all parallel threads.
        if not CHECKPOINT_INITIALIZED:
            for port in ports:
                sd_api.load_checkpoint(port, checkpoint_type)
            CHECKPOINT_INITIALIZED = True  # Mark as initialized


def run_stylus(prompt: str,
               args,
               rank_cache: Optional[Dict[str, List[AdapterInfo]]] = None,
               adapters_cache=None):
    """
    Main function to run the Stylus workflow, managing the entire process from computing rankings to generating images.

    Parameters:
    - prompt (str): The text input used to guide the image generation process.
    - args: The configuration settings or parameters needed to control the image generation. (see configs/default_config.yaml)
    - rank_cache (Dict[str, List['AdapterInfo']], optional): A cache of previously computed rankings of adapters to speed up the process. Defaults to None.
    - adapters_cache: A cache for all adapter dataset (see cache/sd_adapters.pkl) to prevent redundant data retrieval across threads. Defaults to None.
    """
    output_dir = os.path.abspath(os.path.expanduser(args.output_dir))
    sd_api.set_available_ports(args.sd_config.ports)
    cur_file_path = os.path.abspath(__file__)
    lora_cache = os.path.join(os.path.dirname(cur_file_path),
                              SD_WEBUI_LORA_FOLDER)

    # Compute Ranks.
    if rank_cache:
        print("Skipping Stylus, using cached adapter rankings.")
        final_loras = rank_cache
    else:
        # ============================================================#
        # Stylus Retriever - RAG for top K adapters.
        # ============================================================#
        with Timer('Stylus Retriever - RAG for top K adapters'):
            top_loras = compute_rankings(
                prompt=prompt,
                top_k=args.rank.top_k,
                policy=args.rank.policy,
                embedding_type=args.rank.embedding_type,
                adapters_cache=adapters_cache,
                pinecone=args.rank.pinecone)

        if args.rerank.enable:
            # ============================================================#
            # Stylus Composer - Reranks adapters + align with keywords.
            # ============================================================#
            if not top_loras:
                final_loras = {}
            elif args.rerank.policy == 'cohere':
                # This is for our ablations against existing rerankers, such as Cohere (SoTA).
                from stylus.composer.reranker import rerank
                rerank_loras = rerank(prompt=prompt, top_adapters=top_loras)
                final_loras = {
                    str(i): [rerank_loras[i]]
                    for i in range(args.rerank.top_k)
                }
            elif args.rerank.policy == 'composer':
                from stylus.composer.composer import compose
                with Timer(
                        'Stylus Composer - Aligning top adapters with prompt'):
                    final_loras = compose(
                        prompt=prompt,
                        adapters=top_loras,
                        top_k=args.rerank.top_k,
                        num_concepts=args.rerank.num_concepts,
                        rerank_model=args.rerank.rerank_model,
                        enable_characters=not args.rerank.block_characters,
                        enable_multiturn=args.rerank.multiturn,)
            else:
                raise ValueError(
                    f"Invalid rerank policy: {args.rerank.policy}")
        else:
            # Fetch top 3 from RAG if rerank is disabled.
            final_loras = {
                str(i): [top_loras[i]]
                for i in range(args.rerank.top_k)
            }
    if args.skip_generation:
        return final_loras

    # ============================================================#
    # Download/Fetch Adapters
    # ============================================================#
    with Timer('Downloading/Fetch Adapters'):
        download_loras(final_loras, lora_cache)

    # ============================================================#
    # Generate Masks
    # ============================================================#
    masks = get_masks(final_loras, strategy=args.mask)

    # ============================================================#
    # Generate Stylus Images
    # ============================================================#
    load_checkpoints(ports=args.sd_config.ports,
                     checkpoint_type=args.sd_config.checkpoint)

    # Stylus batch size logic.
    batch_size_per_mask = args.sd_config.batch_size_per_mask
    batch_size = args.sd_config.batch_size

    if batch_size_per_mask is None or batch_size_per_mask <= 0:
        # sample Masks with replacement.
        indices = list(range(len(masks)))
        selected_indices = random.choices(indices, k=batch_size)

        set_indices = list(set(selected_indices))
        final_masks = [masks[i] for i in set_indices]
        mask_counts = Counter(selected_indices)
        mask_batch_sizes = [mask_counts[idx] for idx in set_indices]
    else:
        mask_batch_sizes = [batch_size_per_mask] * len(masks)
        final_masks = masks
    assert len(mask_batch_sizes) % len(final_masks) == 0, "Oops"
    fname = prompt_to_filename(prompt)
    counter = 0
    with Timer(f'Generating Stylus Images'):
        for b_size, mask in zip(mask_batch_sizes, final_masks):
            lora_prompt = prompt_builder(
                prompt,
                final_loras,
                bias_str=args.debias_str,
                restore_concepts=args.restore_concepts,
                mask=mask)
            print(lora_prompt)
            generate_from_prompt(lora_prompt,
                                 args.sd_config,
                                 output_dir,
                                 fname,
                                 with_lora=True,
                                 ignore_cache= counter > 0,
                                 batch_size=b_size)
            counter+=1

    # ============================================================#
    # Generate Normal Images
    # ============================================================#
    with Timer(f'Generating Normal Images'):
        counter = 0
        for b_size, mask in zip(mask_batch_sizes, final_masks):
            prompt = prompt.lower()
            print(prompt)
            generate_from_prompt(prompt,
                                args.sd_config,
                                output_dir,
                                fname,
                                ignore_cache= counter > 0,
                                batch_size=b_size,)
            counter+=1
    return final_loras
