import requests
import io
import base64
from PIL import Image
from copy import deepcopy
import random
import threading

CHECKPOINT_DICT = {
    'base': 'v1-5-pruned-emaonly.safetensors',
    'realistic-vision': 'realisticVisionV60B1_v51VAE.safetensors',
    'counterfeit': 'counterfeitV30_v30.safetensors',
}
RR_LOCK = threading.Lock()
TXT2IMG_DEFAULT = {
    "prompt": "",
    "negative_prompt": "",
    "styles": [],
    "seed": -1,
    "subseed": -1,
    "subseed_strength": 0,
    "seed_resize_from_h": -1,
    "seed_resize_from_w": -1,
    "sampler_name": "DPM++ 2M Karras",
    "batch_size": 1,
    "n_iter": 1,
    "steps": 20,
    "cfg_scale": 7,
    "width": 512,
    "height": 512,
    "restore_faces": None,
    "tiling": None,
    "do_not_save_samples": False,
    "do_not_save_grid": False,
    "eta": None,
    "denoising_strength": 0,
    "s_min_uncond": 0,
    "s_churn": 0,
    "s_tmax": 0,
    "s_tmin": 0,
    "s_noise": 0,
    "override_settings": {},
    "override_settings_restore_afterwards": True,
    "refiner_checkpoint": None,
    "refiner_switch_at": None,
    "disable_extra_networks": False,
    "comments": {},
    "enable_hr": False,
    "firstphase_width": 0,
    "firstphase_height": 0,
    "hr_scale": 2,
    "hr_upscaler": "Latent",
    "hr_second_pass_steps": 0,
    "hr_resize_x": 0,
    "hr_resize_y": 0,
    "hr_checkpoint_name": None,
    "hr_sampler_name": None,
    "hr_prompt": "",
    "hr_negative_prompt": "",
    "sampler_index": "Euler",
    "send_images": True,
    "save_images": False,
    "alwayson_scripts": {}
}

NEGATIVE_PROMPTS = [
    "realisticvision-negative-embedding",
    "ng_deepnegative_v1_75t",
    "bad anatomy",
    "bad proportions",
    "blurry",
    "cloned face",
    "cropped",
    "deformed",
    "dehydrated",
    "disfigured",
    "duplicate",
    "error",
    "extra arms",
    "extra fingers",
    "extra legs",
    "extra limbs",
    "fused fingers",
    "gross proportions",
    "jpeg artifacts",
    "long neck",
    "(low quality: 2)",
    "lowres",
    "malformed limbs",
    "missing arms",
    "missing legs",
    "morbid",
    "mutated hands",
    "mutation",
    "mutilated",
    "out of frame",
    "poorly drawn face",
    "poorly drawn hands",
    "signature",
    "text",
    "too many fingers",
    "ugly",
    "username",
    "watermark",
    "(worst quality:2)",
]
NEGATIVE_PROMPT_STR = ", ".join(NEGATIVE_PROMPTS)
WEBUI_URL = "http://127.0.0.1"
AVAILABLE_PORTS = [7860]
RR_INDEX = 0


def set_available_ports(ports):
    global AVAILABLE_PORTS
    AVAILABLE_PORTS = ports


def get_port(policy="round_robin", ports=AVAILABLE_PORTS):
    if policy == "round_robin":
        with RR_LOCK:
            global RR_INDEX
            selected_port = ports[RR_INDEX]
            RR_INDEX = (RR_INDEX + 1) % len(ports)
        return selected_port
    elif policy == "random":
        return random.choice(ports)
    elif policy == "first":
        return ports[0]
    else:
        raise NotImplementedError


def txt2img(prompt: str,
            negative_prompt: str = NEGATIVE_PROMPT_STR,
            steps=20,
            batch_size: int = 1,
            seed=-1,
            subseed=-1,
            subseed_strength=0,
            cfg_scale=7,
            enable_hr=True,
            denoising_strength=0.7,
            hr_second_pass_steps=0,
            port=7860):
    # Populate Rest API parameters.
    payload = deepcopy(TXT2IMG_DEFAULT)
    payload['prompt'] = prompt
    payload['seed'] = seed
    payload['batch_size'] = batch_size
    payload['negative_prompt'] = negative_prompt
    payload['steps'] = steps
    payload['subseed'] = subseed
    payload['subseed_strength'] = subseed_strength
    payload['cfg_scale'] = cfg_scale
    payload['enable_hr'] = enable_hr
    payload['denoising_strength'] = denoising_strength
    payload['hr_second_pass_steps'] = hr_second_pass_steps
    # Send request to Rest API.
    response = requests.post(url=f'{WEBUI_URL}:{port}/sdapi/v1/txt2img',
                             json=payload)
    r = response.json()
    assert len(
        r['images']
    ) == batch_size, f"Expected {batch_size} images, but got {len(r['images'])}."
    images = [
        Image.open(io.BytesIO(base64.b64decode(i.split(",", 1)[0])))
        for i in r['images']
    ]
    return images


def refresh_loras(port):
    response = requests.post(f"{WEBUI_URL}:{port}/sdapi/v1/refresh-loras")
    return response.json()


def load_checkpoint(port: int, checkpoint_type: str):
    if checkpoint_type not in CHECKPOINT_DICT:
        checkpoint_type = 'realistic-vision'
    checkpoint = CHECKPOINT_DICT[checkpoint_type]
    payload = {'sd_model_checkpoint': checkpoint}
    response = requests.post(f"{WEBUI_URL}:{port}/sdapi/v1/options",
                             json=payload)
    return response.json()
