# Generating StylusDocs

This tutorial assumes that the user has created an account on Civit AI and has generated an API key. To generate your own StylusDocs, we provide 4 steps:

## 1. Civit AI Catalog.

The first step is to fetch all of Civit AI's models via Civit's Rest API. This is saved in `stylus/cache/civit_catalog.json`. Run:
```
python fetch_catalog.py
```

This should take 1-2 hours to complete. If not, the API server is congested.

## 2. Adapter Metadata

The second steps involves extracting metadata from the Civit AI catalog and, for each adapter, extracting images, prompts, and negative prompts from Civit AI's Rest API.
This is saved in `stylus/cache/sd_adapters.pkl`. Run:
```
python fetch_adapter_metadata.py
```

## 3. Added VLM descriptions to Adapter Metadata

The refiner employs a VLM (Gemini-1.5 or GPT-4o) to improve the descriptions and documentations for each adapter. Our code assumes that the user has a GCP (Google Cloud) or OpenAI account setup.
The descriptions of each adapter is saved in `stylus/cache/ultra_model/[MODEL_ID]`, where `MODEL_ID` is the Civit AI model ID. Run:
```
# for gemini
python vlm.py --vlm gemini

# for openai
python vlm.py --vlm openai
```
We recommend running this several times to ensure that most adapters have VLM descriptions. The length of this may take hours to weeks, depending on the user's GCP quota for Gemini.

## 4. Convert Adapter into Embedding

This refiner takes in VLM's description (or regular model card description if no VLM description is provided) and converts it into a text embedding.
By default, we rely on OpenAI's text embedding service. This is saved in `stylus/cache/sd_embeddings.npy`. Run:
```
python encoder.py
```