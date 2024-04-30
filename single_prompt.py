"""
Script to run a single prompt with Stylus.

Examples:
# Run Stylus with a single prompt.
# NOTE: Default config does not apply masking. Hence, all adapters are used.
> python single_prompt.py --prompt "a dragon breathing fire on a castle"

# Run Stylus with a single prompt with a preetermined set of adapter IDs. (civit.ai/models/[ID])
> python single_prompt.py --prompt "A pair of cows socialize in a field." --rank_cache '{"cows": [99911], "field": [345241]}'
"""
import argparse
import ast

from stylus.utils.config import load_config
from stylus.launch_stylus import run_stylus
from stylus.retriever.rag import get_all_adapters


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run Stylus with a single prompt.')
    parser.add_argument('--prompt', '-p', type=str, help='User prompt.')
    parser.add_argument('--config',
                        type=str,
                        help='Stylus config file (see configs/)',
                        default='configs/default_config.yaml')
    parser.add_argument('--rank_cache', type=str, help='JSON string of ranked IDs', default='{}')

    args = parser.parse_args()
    prompt = args.prompt
    full_args = load_config(args.config)
    ranked_id = args.rank_cache
    ranked_id = ast.literal_eval(ranked_id)
    assert isinstance(ranked_id, dict), 'Ranked cache must be a dictionary.'

    rank_cache= None
    if ranked_id:
        all_adapters = get_all_adapters()
        all_adapters_lookup = {
            adapter.adapter_id: adapter
            for adapter in all_adapters
        }
        rank_cache = ({
            concept: [
                all_adapters_lookup[idx] for idx in ranked_id[concept]
                if idx in all_adapters_lookup
            ]
            for concept in ranked_id.keys()})
    run_stylus(prompt=prompt, args=full_args, rank_cache=rank_cache)
