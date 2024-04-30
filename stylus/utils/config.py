import os
from omegaconf import OmegaConf
from dotenv import load_dotenv

load_dotenv()


def set_env_keys(cfg):
    """
        Load keys from keys.yaml if missing look for environment variables
    """
    keys_cfg = OmegaConf.load(cfg.keys_path)

    # CIVITAI
    if keys_cfg.CIVIT_API_KEY is not None:
        if os.getenv("CIVIT_API_KEY") != keys_cfg.CIVIT_API_KEY:
            print(
                f"Warning: CIVIT_API_KEY is set in keys.yaml conflicts with environment variable. Overwriting environment variable.\nENV:{os.getenv('CIVIT_API_KEY')}\nkeys.yaml:{keys_cfg.CIVIT_API_KEY}"
            )
            os.environ["CIVIT_API_KEY"] = keys_cfg.CIVIT_API_KEY
    else:
        if not os.environ.get('CIVIT_API_KEY'):
            raise ValueError(
                'CIVIT_API_KEY environment variable is not set. Please fetch your API key from https://civitai.com and run `export CIVIT_API_KEY=[KEY]`.'
            )

    # OPENAI
    if keys_cfg.OPENAI_API_KEY is not None:
        if os.getenv("OPENAI_API_KEY") != keys_cfg.OPENAI_API_KEY:
            print(
                f"Warning: OPENAI_API_KEY is set in keys.yaml conflicts with environment variable. Overwriting environment variable.\nENV:{os.getenv('OPENAI_API_KEY')}\nkeys.yaml:{keys_cfg.OPENAI_API_KEY}"
            )
            os.environ["OPENAI_API_KEY"] = keys_cfg.OPENAI_API_KEY
    else:
        if os.getenv("OPENAI_API_KEY") is None:
            raise ValueError(
                'OPENAI_API_KEY not found in environment nor keys.yaml. Please fetch your API key from https://openai.com and add your key to keys.yaml.'
            )
    if keys_cfg.OPENAI_ORG_ID is not None:
        os.environ["OPENAI_ORG_ID"] = keys_cfg.OPENAI_ORG_ID
    if keys_cfg.OPENAI_PROJECT_ID is not None:
        os.environ["OPENAI_PROJECT_ID"] = keys_cfg.OPENAI_PROJECT_ID

    # PINECONE
    if keys_cfg.PINECONE_KEY is not None:
        if os.getenv("PINECONE_KEY") != keys_cfg.PINECONE_KEY:
            print(
                f"Warning: PINECONE_KEY is set in keys.yaml conflicts with environment variable. Overwriting environment variable.\nENV:{os.getenv('PINECONE_KEY')}\nkeys.yaml:{keys_cfg.PINECONE_KEY}"
            )
            os.environ["PINECONE_KEY"] = keys_cfg.PINECONE_KEY
    else:
        if os.getenv("PINECONE_KEY") is None:
            print(
                'PINECONE_KEY not found in environment nor keys.yaml. Please add your key to keys.yaml.'
            )
    
    # COHERE
    if keys_cfg.COHERE_API_KEY is not None:
        if os.getenv("COHERE_API_KEY") != keys_cfg.COHERE_KEY:
            print(
                f"Warning: COHERE_API_KEY is set in keys.yaml conflicts with environment variable. Overwriting environment variable.\nENV:{os.getenv('COHERE_API_KEY')}\nkeys.yaml:{keys_cfg.COHERE_KEY}"
            )
            os.environ["COHERE_API_KEY"] = keys_cfg.COHERE_KEY
    else:
        if os.getenv("COHERE_API_KEY") is None:
            print(
                'COHERE_API_KEY not found in environment nor keys.yaml. Please add your key to keys.yaml.'
            )


def load_config(config: str, base_config="configs/default_config.yaml"):
    base_cfg = OmegaConf.load(base_config)
    cfg = OmegaConf.load(config)
    final_cfg = OmegaConf.merge(base_cfg, cfg)
    set_env_keys(final_cfg)
    return final_cfg
