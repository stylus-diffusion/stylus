from copy import copy

from stylus.utils import ADAPTER_BLACKLIST


def apply_mask_to_dict(data_dict: dict, mask_dict: dict) -> dict:
    """
    Applies a given mask to the original dictionary, filtering elements based on the mask.

    Parameters:
    - data_dict (dict): The original dictionary containing lists of numbers.
    - mask_dict (dict): A dictionary containing masks to be applied.

    Returns:
    dict: A dictionary with the same structure as data_dict, where each list is filtered by the corresponding mask.
    """
    filtered_dict = {}
    for key in data_dict:
        if key in mask_dict:
            # Apply the mask: keep the element if the mask at the same position is True
            filtered_list = [
                value for value, keep in zip(data_dict[key], mask_dict[key])
                if keep
            ]
            filtered_dict[key] = filtered_list
        else:
            # If no mask is provided for a key, copy the list as is
            filtered_dict[key] = copy(data_dict[key])
    return filtered_dict


def prompt_builder(prompt: str,
                   ranked_loras: dict,
                   bias_str: str = None,
                   mask: dict = None,
                   restore_concepts: bool = False,
                   base_weight: float = 0.8):
    """
    Builds a prompt by applying bias and filtering out unwanted LoRA adapters.

    Parameters:
    - prompt (str): The initial prompt text.
    - ranked_loras (dict): Dictionary of ranked LoRA concepts and their ost list of adapters adapters.
    - bias_str (str, optional): A string to bias the prompt towards the checkpoint style.
    - mask (dict, optional): A dictionary mask to filter out certain loras.

    Returns:
    str: The modified prompt incorporating the adjustments and biases.
    """
    lora_prompt = copy(prompt).lower()
    # The debias string shifts the adapter bias back to the checkpoint bias.
    # For example, Realistic Vision checkpoint follows a realistic bias.
    if bias_str:
        lora_prompt += f', {bias_str}'
    if mask:
        ranked_loras = apply_mask_to_dict(ranked_loras, mask)
    # Optional: Emphasize key words that aren't covered by LORAs.
    # LoRAs can block certain concepts in the image, so we emphasize such concepts
    # to restore them in the image.
    if restore_concepts:
        for concept, loras in ranked_loras.items():
            if not loras:
                lora_prompt = lora_prompt.replace(concept, f'({concept}:1.1)')

    for concept, loras in ranked_loras.items():
        nonblacklisted_loras = [
            lora for lora in loras if lora.adapter_id not in ADAPTER_BLACKLIST
        ]
        num_loras = len(nonblacklisted_loras)
        # NOTE: We do not add trigger word to the prompt. The prompt should already
        # have related concepts/keywords that activate the adapter/LoRA.
        for lora in nonblacklisted_loras:
            # Note that lora.weight is extracted by Refiner's VLM.
            weight = lora.weight * base_weight
            lora_prompt += f', <lora:{lora.adapter_id}:{weight/float(num_loras)}>'
    return lora_prompt
