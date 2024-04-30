def prompt_to_filename(prompt):
    if not isinstance(prompt, str):
        prompt = prompt['caption']
    prompt = prompt.replace(" ", "_")
    prompt = ''.join(e for e in prompt if (e.isalnum() or e == "_"))
    return prompt[:100]
