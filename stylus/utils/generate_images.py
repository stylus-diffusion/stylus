import os

from omegaconf import OmegaConf

from stylus.utils import sd_api


def generate_from_prompt(prompt: str,
                         sd_config: OmegaConf,
                         output_dir: str,
                         fname: str,
                         with_lora: bool = False,
                         ignore_cache=False,
                         batch_size=-1):
    """Generates images from a prompt using the specified configuration."""
    if batch_size == -1:
        batch_size = sd_config.batch_size

    exp_str = "lora" if with_lora else "normal"
    save_dir = os.path.join(output_dir, f'cfg_{sd_config.cfg}', fname, exp_str)
    if os.path.exists(save_dir) and len(os.listdir(save_dir)) > 1 and not ignore_cache:
        print(f"Skipping {save_dir} as it already exists")
        return

    port = sd_api.get_port(policy="round_robin", ports=sd_config.ports)
    print(f"Generating with port {port}")
    sd_api.refresh_loras(port)

    images = sd_api.txt2img(prompt=prompt,
                            batch_size=batch_size,
                            steps=sd_config.steps,
                            cfg_scale=sd_config.cfg,
                            seed=sd_config.seed,
                            subseed=sd_config.subseed,
                            subseed_strength=sd_config.subseed_strength,
                            port=port)

    img_counter = 0
    for s, image in enumerate(images):
        image_path = os.path.join(save_dir, f"{img_counter}_{fname}.png")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        while os.path.exists(image_path):
            img_counter += 1
            image_path = os.path.join(save_dir, f"{img_counter}_{fname}.png")
        image.save(image_path)
