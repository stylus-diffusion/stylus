from itertools import product
import random


def _cross_product_all_keys(data_dict):

    def generate_cross_product_masks(data_dict):
        result_masks = {}
        for key, values in data_dict.items():
            if not values:  # If the list is empty, skip
                continue
            # All possible boolean values for the length of the list
            all_combinations = list(product([False, True], repeat=len(values)))
            # Convert each combination to the required format (list of True/False)
            masks = [list(combination) for combination in all_combinations]
            result_masks[key] = masks
        return result_masks

    # Generate masks for each key
    individual_masks = generate_cross_product_masks(data_dict)

    # Extract keys and corresponding masks
    keys = list(individual_masks.keys())
    mask_lists = [individual_masks[key] for key in keys]

    # Compute the cross product of all mask lists
    cross_product = list(product(*mask_lists))

    # Format the result as a list of dictionaries
    formatted_result = []
    for combination in cross_product:
        combination_dict = {key: mask for key, mask in zip(keys, combination)}
        formatted_result.append(combination_dict)

    return formatted_result


def get_masks(ranked_loras, strategy="one_hot_loras"):
    if strategy == "one_hot_loras":
        all_masks = _cross_product_all_keys(ranked_loras)
        return all_masks
    elif strategy == "one_hot_concepts":
        # Filter out concepts with no LoRAs
        non_empty_concepts = {
            k: v
            for k, v in ranked_loras.items() if len(v) > 0
        }
        concept_keys = list(non_empty_concepts.keys())
        # Generating all combinations of True/False for the non-empty concepts
        all_combinations = list(
            product([True, False], repeat=len(concept_keys)))
        # Convert each combination into a mask
        masks = []
        for combination in all_combinations:
            mask = {
                concept: [state] * len(ranked_loras[concept])
                for concept, state in zip(concept_keys, combination)
            }
            masks.append(mask)
        return masks
    elif strategy == 'all':
        # return one mask, which is all LoRAs.
        return [{
            concept: [True] * len(ranked_loras[concept])
            for concept in ranked_loras
        }]
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


if __name__ == "__main__":
    lists = get_masks({
        "A": [1],
        "B": [0, 1, 2],
        "C": [0, 1]
    },
                      12,
                      strategy="one_hot_loras")
    print(len(lists))
    print(lists)
