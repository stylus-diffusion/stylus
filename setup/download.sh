#!/bin/bash
# Get the directory where the script is located
STYLUSDIR=$(dirname "$(readlink -f "$0")")/..

# =================== SD v1.5 Checkpoints =================== #
# Change to the Stable-diffusion model directory
cd "$STYLUSDIR/stable_diffusion/models/Stable-diffusion"

# Download Realistic Vision model
wget --content-disposition https://civitai.com/api/download/models/130072?type=Model&format=SafeTensor&size=full&fp=fp16

# Download Counterfeit model
wget --content-disposition https://civitai.com/api/download/models/57618?type=Model&format=SafeTensor&size=pruned&fp=fp16

# =================== SD v1.5 Negative Embeddings =================== #
# Change to the embeddings directory
cd "$STYLUSDIR/stable_diffusion/embeddings"

# Download NG DeepNegative
wget --content-disposition https://civitai.com/api/download/models/5637

# Download Realistic Vision Negative Embedding
wget --content-disposition https://civitai.com/api/download/models/42247?type=Model&format=Other


# =================== (Optional) StylusDocs Dataset =================== #
cd ~/
GDRIVE_ID="1iMYQWHraC1JT78-MYOapvGNIuLoTMO-x"
DEST_FOLDER="$STYLUSDIR/stylus/"
gdown --id $GDRIVE_ID -O temp.zip
# Unzip the file into the destination folder
unzip -o temp.zip -d "${DEST_FOLDER}"
# Remove the temporary zip file
rm temp.zip