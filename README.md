# 🖌️ Stylus: Automatic Adapter Selection for Diffusion Models

<p align="center">
  <a href="https://stylus-diffusion.github.io/"><img src="https://img.shields.io/badge/🌐-Website-red" height="25"></a>
  <a href="https://arxiv.org/abs/2404.18928"><img src="https://img.shields.io/badge/📝-Paper-blue" height="25"></a>
  <a href="[https://drive.google.com/file/d/1iMYQWHraC1JT78-MYOapvGNIuLoTMO-x/view?usp=sharing](https://drive.google.com/file/d/1qFEoDWp3BSwyIlkaSrEoRCmMNGogMyVn/view?usp=sharing)" ><img src="https://img.shields.io/badge/💡-StylusDocs-green" height="25"></a>
</p>

** Update: Stylus has won an Oral award for NeurIPS 2024! **

** Update: We have uploaded [StylusDocsv2](https://drive.google.com/file/d/1qFEoDWp3BSwyIlkaSrEoRCmMNGogMyVn/view?usp=sharing), with better adapter descriptions from GPT-4o. **

## 🌎 Overview

Stylus automatically retrieves and composes relevant adapters based on prompts' keywords, generating beautiful and creative images that are tailor-made for each user.

<p align="center">
  <img src="stylus_examples/stylus-gif-final.gif" width="600">
</p>

## 🔧 Setup

We recommend using Python version `>=3.10`. To install Stylus, run:
```
git clone --recursive https://github.com/stylus-diffusion/stylus.git
cd stylus
pip install -r requirements.txt
pip install -e .
```

To download the neccessary checkpoints, embeddings for Stable Diffusion WebUI and StylusDocs (our dataset), run:
```
./setup/download.sh
```

Finally, Stylus requires API keys from OpenAI ([link](https://platform.openai.com/api-keys)), Civit AI ([link](https://developer.civitai.com/docs/getting-started/setup-profile)).
Place the keys in `configs/keys.yaml` and Stylus will automatically manage the env variables for you. 

Alternatively, such env variables can also be added to `~/.bashrc` or `~/.zshrc`.
```
echo 'export CIVIT_API_KEY="[CIVIT_KEY]"' >> ~/.bashrc
echo 'export OPENAI_API_KEY="[OPENAI_KEY]"' >> ~/.bashrc
```

NOTE: Stylus also supports Google's Gemini 1.5 for Composer component and Pinecone DB for the Retriever component. Login to your GCP cloud account (`gcloud init`) to access Gemini 1.5 and add the Pinecone key to `configs/key.yaml` for such options to work.

## 🚀 Launch Stylus

Stylus can be launched in two simple steps. The first step is launching Stable Diffusion Web UI ([link](https://github.com/AUTOMATIC1111/stable-diffusion-webui)).

First, to launch SD Web UI, run:
```
# NOTE: We highly recommend running this in a separate Python environment, due to conflicting dependencies with Stylus.
conda create -n sd python=3.10
conda activate sd
# Launches N replicas of SD Web UI on ports 7860, ..., 7860 + N-1
./setup/launch_sd.sh [N]
```

Second, Stylus can be run in two ways:
```
# For a single prompt. See `configs/default_config.yaml` for default options.
python single_prompt.py --prompt "[PROMPT]" --config [CONFIG_PATH]

# For a batch of prompts, we require them to be in a CSV file (see `datasets/`)
python batch_prompts.py --config [CONFIG_PATH]
```

Both programs will output images generated by Stylus (lora) and the base checkpoint (normal) in the directory: `[OUTPUT_FOLDER]/cfg_[CFG]/{lora/normal}/[PROMPT]/...` See examples of Stylus generated images in `stylus_examples/`.

## 📝 Miscellaneous

### Creating StylusDocs (Refiner)

We provide a step by step tutorial for re-creating StyusDocs in `stylus/refiner/README.md`.

### Experiments

See `stylus/eval/README.md` to run and plot our experiments in the paper. The plotting code and corresponding figures are located in `plots/`.

## 🎯 Citation

```
@misc{luo2024stylus,
      title={Stylus: Automatic Adapter Selection for Diffusion Models}, 
      author={Michael Luo and Justin Wong and Brandon Trabucco and Yanping Huang and Joseph E. Gonzalez and Zhifeng Chen and Ruslan Salakhutdinov and Ion Stoica},
      year={2024},
      eprint={2404.18928},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

