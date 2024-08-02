import ast
import time
from typing import List

from google.cloud.aiplatform_v1beta1.types.content import SafetySetting
from openai import OpenAI
import vertexai
from vertexai.generative_models import GenerativeModel, HarmBlockThreshold

from stylus.refiner.fetch_adapter_metadata import AdapterInfo
from stylus.retriever.rag import compute_rankings
from stylus.refiner.vlm import HARM_CATEGORIES
from stylus.utils import blacklist_adapters


# Template strings for Composer prompts.
ADAPTER_CATALOG = """\
Index: {adapter_idx}
Title: {adapter_title}
Tags: {adapter_tags}
Description: {adapter_description}
=========================================\n
"""
COMPOSER_PROMPT = """\
{adapters_catalog_str}
Provided above are the descriptions for different model adapters (LoRA) that may be related to the prompt. The prompt is:
"{prompt}".

Your goal is to find LoRA(s) that can improve image quality and more strictly ahere to the prompt. 

First, segment the prompt into different topics - such as concepts/styles/poses/celebrities/backgrounds/objects/actions/adjectives - among the key words in the prompt. Identify all topics, which must cover all the keywords in the entire prompt. Partition the prompt into as many topics as possible.

Here are the requirements for topics:
0) Split the sentence into key words. These are the topics. Topics can also be adjectives. Be finegrained in this.
1) These topics should never introduce new information to the prompt. The topic must be selected from the words from the prompt.
2) Different topics must be orthagonal (completely separate) from each other.
3) Prioritize choosing narrower topics. You may merge topics together if needed.
{limit_concepts_str}.

Second, for each topic, provide 0-{top_k} of the most relevant model adapters to the topic and prompt by their index.

Here are the requirements for adapters per topic:
1) Avoid adapters that are potentially sexually explicit in content!
2) Avoid anthropormorphic adapters at all costs, such as catgirls or animal-inspired humanoids.
3) Adapters should only be used at MOST ONCE across all topics. If an adapter is used in one topic, it should not be used in another topic.
4) Infer an adapter's main purpose/functionality. The adapter's main purpose must absolutely and directly match the best-fitting topic. The purpose must also match the context of the prompt. If the adapter's main purpose is not relevant to the prompt, do not include it.
5) Adapters cannot be a superset or a broader scope than the topic.  These adapters should never introduce new things to the topics and the prompt. Most importantly, the adapters must perfectly fit the context of the prompt and the topic.  If not, do not include it. For example, if the topic is "hot dog", the adapter cannot be about "general food".
6) Adapters cannot be too narrow in scope relative to the topic. Prioritize choosing adapters that perfectly match the topic before choosing narrower adapters. For example if a topic is an animal, such as "dog", do not give adapters about a dog pet's name or a famous dog character/celebrity such as "Bolt from pixar" or "Clifford the big red dog". These are too specific for dogs; however, "German Shepherd" or "Golden Retriever" are acceptable, as these correspond to breeds of dogs. For example, If the topic is about panda, but the adapter's main purpose is to generate a character Po from Kung Fu Panda, do not include the adapter. You may also return adapters that can modify the style of the topic. Such as for bathroom - modern bathroom, vintage bathroom, etc.
7) Adapters must strictly adhere with the topic and fall within the same type of topic, i.e. concepts/styles/poses/celebrities/backgrounds/objects/actions. If no LORAs fit, this is perfectly fine to return 0 adapters for a topic.
8) If the topic is not a character or celebrity, DO NOT include adapters that are about a specific character or celebrity. For example, topics such as "man" or "woman" or "girl" or "boy" are not characters or celebrities - they should not be attached to specific characters or celebrity adapters (incdicated by the character/celebrity tags).
9) If an adapter's purpose somehow spans multiple topics, merge the topics together. For example, if there is an adapter that is about "black cats", merge the topics "black" and "cats" together.

Give me the answer only.

The output format should be in json format with the following keys and values:
```
{{
\"[topic_name]\": {{
\"[adapter index]\": \"[Describe the adapter's primary function. Then describe how this function directly matches the topic name and the context of the prompt. Do not hallucinate. If it IS NOT A PERFECT MATCH, DO NOT CHOOSE THE ADAPTER. DO NOT ACCEPT ADAPTERS THAT ARE SIMILAR TO THE TOPIC OR COULD INDIRECTLY IMPACT THE TOPIC OR THAT MIGHT CONTAIN THE TOPIC. Output at most two style LoRAs for the entire output. Then, give a strong reason how the adapter's primary purpose can improve the image generation quality for the prompt.]\",
...
}},
...
}}
```

For example, the prompt "James bond covered in blood in Russia" with topics, "James bond" and "blood", found with adapters. The output could be:

{{
    \"James bond\":
    {{
        \"72\": \"The adapter 'Sean Connery` is about an actor who played James Bond in the 1960s. It matches the topic James Bond. This adapter provides crisper images of Sean Connery to better represent James Bond.\",
        \"81\": \"The adapter 'Daniel Craig` is about an actor who played James Bond in the 2000s. It matches the topic James Bond. This adapter provides crisper images of Daniel Craig to better represent James Bond.\",
    }},
    \"blood\":
    {{
        \"90\": \"The adapter 'blood splatter` is about the style of blood splatter. Blood splatters can increase the detail of blood depicted on James Bond's body\".
    }},
    \"Russia\": {{}},
}}
Another example, for the prompt "a black t-shirt with the peace sign on it", the output should be:

{{
    \"t-shirt\":
    {{
        \"9\": \"This adapter 'T-shirt design' is trained on a dataset of t-shirt designs and can generate new t-shirt designs. This can improve the image generation quality by providing a variety of t-shirt designs.\",
        \"62\": \"This adapter 'T-shirt' is trained on a dataset of t-shirts and can generate new t-shirts. This can improve the image generation quality by generating a wider set of t-shirts.\",
    }},
    \"peace sign\":
    {{
        \"127\": \"This adaper is about the peace sign logo, which can be pasted onto t-shirts. This will provide a clearer image of a peace sign, improving image generation.\",
    }},
    \"black\": {{}},
}}
"""

MULTITURN_PROMPT = 'Refine your answer to fit the json output format. However, make sure that the adapter closely fits the topic and the prompt. REMOVE ADAPTERS THAT ARE "SIMILAR" TO THEIR TOPIC OR COULD INDIRECTLY IMPACT A TOPIC OR COULD INCLUDE THE OBJECT. THE ADAPTER MUST BE DIRECTLY RELATED TO THE TOPIC. Also, Remove adapters that cover a broader topic than its assigned topic. Absolutely make sures that the adapters follow the rules above. Remove all character/celebrity LORAs if there is no celebrity or character in the prompt. Make sure that each adapter is provided AT MOST ONCE across topics and only 0-{top_k} adapters per topic. If an adapter covers multiple topics, group the topics together. Do not hallucinate. Provide only the answer in the output format below:'

class Composer:
    def __init__(self, model: str):
        self.model = model
        
    def forward(self, prompt: str):
        raise NotImplementedError
        
        
class GeminiComposer(Composer):
    def __init__(self, model: str = 'gemini-1.5-pro-001'):
        super().__init__(model)
        vertexai.init()
        self.gemini_model = GenerativeModel(self.model)
        self.safety_settings = [
            SafetySetting(
                category=h,
                threshold=HarmBlockThreshold.BLOCK_NONE,
            ) for h in HARM_CATEGORIES
        ]
        self.chat = self.gemini_model.start_chat(history=[], response_validation=False)
        self.backoff_attempts = 1000
    
    def forward(self, prompt: str):
        if not prompt:
            return None
        counter = 0
        timeout = 4
        for _ in range(self.backoff_attempts):
            try:
                res = self.chat.send_message(prompt,
                    safety_settings=self.safety_settings)
                # This will raise error if response is not valid.
                res.text
                return res.text
            except Exception as e:
                print(e)
                if 'list index' in str(e) or 'Internal error' in str(e):
                    return None
                time.sleep(timeout)            
        return None

class OpenAIComposer(Composer):
    def __init__(self, model: str = 'gpt-4o'):
        super().__init__(model)
        self.client = OpenAI()
        self.messages =  [{
            "role": "system",
            "content":
            "You are a recommender system that recommends adapters from a database of millions of model adapters for popular base models such as Stable Diffusion, SDXL, and LLama."
        }]
    
    def forward(self, prompt: str):
        if not prompt:
            return None
        self.messages.append({"role": "user", "content": prompt})
        response = self.client.chat.completions.create(model=self.model, messages=self.messages)
        response_str = response.choices[0].message.content
        self.messages.append({
            "role": "assistant",
            "content": response_str,
        })
        return response_str


def generate_adapters_catalog(adapters: List[AdapterInfo]):
    """Generates the list of adapter descriptions for the Composer's prompt."""
    adapter_catalog_str = ""
    for idx, adapter in enumerate(adapters):
        if adapter.llm_description:
            adapter_description = adapter.llm_description
        else:
            adapter_description = adapter.description[:1000] if adapter.description else 'None'
        
        adapter_catalog_str += ADAPTER_CATALOG.format(adapter_idx=idx,
                                                        adapter_title=adapter.title,
                                                        adapter_tags=adapter.tags,
                                                        adapter_description=adapter_description)
    return adapter_catalog_str

def convert_response_str_to_dict(response_str: str):
    response_str = response_str[response_str.find('{'):response_str.rfind('}') + 1]
    response_dict = ast.literal_eval(response_str)
    return response_dict

def compose(prompt: str,
            adapters: List[AdapterInfo],
            top_k=3,
            num_concepts: int = -1,
            rerank_model='openai',
            enable_characters=False,
            enable_multiturn=True):
    # Blacklist filters
    adapters = blacklist_adapters(adapters, enable_characters)
    adapters_catalog_str = generate_adapters_catalog(adapters)
    limit_concepts_str = '' if num_concepts <= 0 else f"4) Output at most {num_concepts} topics."
    composer_prompt = COMPOSER_PROMPT.format(adapters_catalog_str=adapters_catalog_str,
                                          prompt=prompt,
                                          limit_concepts_str=limit_concepts_str,
                                          top_k=top_k)
    refine_prompt = MULTITURN_PROMPT.format(top_k=top_k)

    if rerank_model == 'gemini':
        composer_model = GeminiComposer()
    elif rerank_model == 'openai':
        composer_model = OpenAIComposer()
    response_str = composer_model.forward(composer_prompt)
    if enable_multiturn:
        response_str = composer_model.forward(refine_prompt)
    if not response_str:
        return {}

    # Convert composer response string to dict.
    response_dict = convert_response_str_to_dict(response_str)
    adapter_concepts_dict = {}
    seen_ids = []
    for concept, adapter_dict in response_dict.items():
        # Sus edge case.
        if concept.lower() == 'prompt':
            continue
        print(f"Concept: {concept}")
        if concept not in adapter_concepts_dict:
            adapter_concepts_dict[concept] = []
        for id, desc in adapter_dict.items():
            if int(id) in seen_ids:
                continue
            seen_ids.append(int(id))
            if int(id) >= len(adapters):
                continue
            le_adapter = adapters[int(id)]
            adapter_concepts_dict[concept].append(le_adapter)
            print(f"\t- {le_adapter.title}: {desc}")
    return adapter_concepts_dict


if __name__ == '__main__':
    prompt = "A man riding a skateboard down a side walk."
    adapters = compute_rankings(prompt=prompt, top_k=200, policy='rank')
    compose(prompt, adapters)
