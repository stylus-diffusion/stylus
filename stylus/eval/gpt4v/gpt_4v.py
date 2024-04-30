# gpt_4v
import base64
import json
import random

import argparse
from openai import OpenAI
import re
from pathlib import Path
from stylus.utils.filename import prompt_to_filename
import pandas as pd
import shutil
import pandas as pd


client = None
random.seed(0)

def get_image_base64(image_path):
    with open(image_path, 'rb') as file:
        return base64.b64encode(file.read()).decode('utf-8')
    
def query_gpt4v(payload) :  
    global client
    if client is None:
        client = OpenAI()
    try:
        completion = client.chat.completions.create(**payload)
        response = completion.choices[0].message.content
        if "cannot assist" in response:
            response = None
        print(response)
    except Exception as e:
        print(f"VLM Error: {e}")
        response = None
    return response

def post_process(response, task="diversity", group_vs_image = "Image"):
    category = get_category(task)
    label = get_label(task).replace(" ", "\\s")
    '''Get the first number after the first colon in the response'''
    all_entries = []
    for line in response.split("\n"):
        pattern = r'Group\s(?P<group>[AB])\s' + category+ r':\s(?P<value>\d+)\s\((?P<explanation>.+?)\)'
        matches = re.findall(pattern, line)
        matches = [{name: val for val,name in zip(m, ["Group", "Score", "Explanation"])} for m in matches]
        for match in matches:
            match["Name"] = f"Group {category}"
        all_entries.extend(matches)
    for line in response.split("\n"):
        pattern = label + r':\s' + group_vs_image + r'\s(?P<choice>[AB])\s\((?P<explanation>.+?)\)'
        matches = re.findall(pattern, line)
        matches = [{name: val for val,name in zip(m, ["Choice",  "Explanation"])} for m in matches]
        for match in matches:
            match["Name"] = "Preference"
        all_entries.extend(matches)
    return all_entries
    

def get_label(task):
    if task == "diversity":
        return "More Diverse"
    elif task == "alignment":
        return "Better Aligned"
    elif task == "quality":
        return "Better Quality"
    else:
        raise NotImplementedError
    
def get_category(task):
    if task == "diversity":
        return "Diversity"
    elif task == "alignment":
        return "Alignment"
    elif task == "quality":
        return "Quality"
    else:
        raise NotImplementedError
def prompt_by_group(task = "diversity", theme = None, COT = False, group_vs_image = "Image"):
        label = get_label(task)
        category = get_category(task)
        text = ""
        
        if task == "alignment":
            assert  theme is not None
            text = f"Both IMAGE A and IMAGE B have the theme: '{theme}'. Remember themes can have several interpretations.\n"
            text = text + f"Rate the {task} of the images in IMAGE A and IMAGE B. For each group, provide a score and explanation. Use the following template: \n\n"
            if COT: 
                text = text + f"Image A: <IMAGE ALIGNMENT SCORE> (<EXPLANATION>)\n\n"
            text = text + f"IMAGE A {category}: <IMAGE SCORE> (<EXPLANATION>)\n\n"
            if COT: 
                text = text + f"Analyze Group B:\nImage 1: <IMAGE ALIGNMENT SCORE> (<EXPLANATION>)\nImage 2: <IMAGE ALIGNMENT SCORE> (<EXPLANATION>)\n...\n\n"
            text = text + f"IMAGE B {category}: <SCORE> (<EXPLANATION>)\n{label}: {group_vs_image} <CHOICE> (<EXPLANATION>)\n\n "
        elif task == "quality":
            text = text + f"Rate the {task} of the images in IMAGE A and IMAGE B. For each image, provide a score and explanation. Use the following template: \n\n"
            text = text + f"IMAGE A {category}: <IMAGE SCORE> (<EXPLANATION>)\n\n"
            text = text + f"IMAGE B {category}: <SCORE> (<EXPLANATION>)\n{label}: {group_vs_image} <CHOICE> (<EXPLANATION>)\n\n "

        else:
            text = text + f"Rate the {task} of the images in GROUP A and GROUP B. For each group, provide a score and explanation. \n\nGroup A {category}: <SCORE> (<EXPLANATION>)\nGroup B {category}: <SCORE> (<EXPLANATION>)\n{label}: {group_vs_image} <CHOICE> (<EXPLANATION>)\n\n. "
        if task == "diversity":
            text = text + "Don't forget we want to reward different main subjects in the diversity score."
            text = text + f"You must pick a group for '{label},' neither is not an option. If it's a close call, make a choice first then explain why in parenthesis. I'll make my own judgement using your results, your response is just an opinion as part of a rigorous process."
        else:
            text = text + f"You must pick an image for '{label},' neither is not an option. If it's a close call, make a choice first then explain why in parenthesis. I'll make my own judgement using your results, your response is just an opinion as part of a rigorous process."
        
        return text
    
def prompt_per_image(group, group_vs_image = "Image"):
    return f"This is {group_vs_image} {group}. Reply 'ACK'."

def judge(directory, base_prompt, lora_is_A, task = "diversity", debug = False):
    num_normal = len(list((Path(directory) / base_prompt / "normal" ).iterdir()))
    num_lora = len(list((Path(directory) / base_prompt / "lora" ).iterdir()))
    num= min(num_normal, num_lora)

    if task == "diversity":
        samples = list(range(0, num))
        if num > 5:
            samples = random.sample(samples, 5)
        img_path_lora = [Path(directory) / base_prompt/"lora"/f"{i}_{base_prompt}.png" for i in samples]
        img_path_normal = [Path(directory) / base_prompt  /"normal" / f"{i}_{base_prompt}.png" for i in samples]
    else:
        samples = list(range(0, num))
        samples = random.sample(samples, 1)
        img_path_lora = [Path(directory) / base_prompt/"lora"/f"{i}_{base_prompt}.png" for i in samples]
        img_path_normal = [Path(directory) / base_prompt  /"normal" / f"{i}_{base_prompt}.png" for i in samples]
        
        if debug:
            debug_dir = Path("./debug") / base_prompt
            debug_dir.mkdir(parents=True, exist_ok=True)
            (debug_dir / 'lora').mkdir(parents=True, exist_ok=True)
            (debug_dir / 'normal').mkdir(parents=True, exist_ok=True)
            if not (debug_dir / "lora" / img_path_lora[0].name ).exists():
                shutil.copy(img_path_lora[0], debug_dir / 'lora')
            if not (debug_dir / "normal" / img_path_lora[0].name ).exists():
                shutil.copy(img_path_normal[0], debug_dir / 'normal')

    base64_image_lora = [get_image_base64(img) for img in img_path_lora]
    base64_image_normal = [get_image_base64(img)for img in img_path_normal]

    # prompt = "How many images are there?" # There are 15 images displayed here. Each image features a young child with different dogs and often wearing various tutus and accessories.
    base64_image_A = base64_image_lora if lora_is_A else base64_image_normal
    base64_image_B = base64_image_normal if lora_is_A else base64_image_lora
    system_prompt_diversity = ("You are a photoshop expert judging which set of images is more diverse. "
                + "\n\nDiversity scores can be 2 (very diverse), 1 (somewhat diverse), 0 (not diverse). "
                + "\nFor instance, the theme 'give an image of it raining cats and dogs' can be interpreted literally as cats and dogs falling from the sky or figuratively as heavy rain. Here the images are diverse cause they show both weather and animals. If the group only contains images of heavy rain, a diversity score of 1 should be given. You will not know the theme."
                + "\nAnother reason for diversity is in the diversity of main subject. In the case where the set contains a mix of images of apples and children dressed as different kinds of apples, this is more diverse than a set with only children dressed as apples. Note the more diverse set has children as the subject in some images and apples as the subject in the others."  
                + "\n\n")
    system_prompt_alignment = ("You are a photoshop expert judging which image follows the theme best. "
                             + "\n\nAlignment scores can be 2 (fully aligned), 1 (incorporates part of the theme but not all), 0 (not aligned). "
                             + "\nFor instance, the theme may be 'give an image of raining cats and dogs'. If the image is of cats and dogs falling from the sky or of heavy rain both get a score of 2. "
                             +"If the image is of a wet towel with images of cats and dogs printed on it. This is scored 1 for containing 'cats' and 'dogs' which are keywords but in context of the theme, it is not aligned. "
                             + "If the prompt is 'shoes', and an image is a sock this is not aligned and gets a score of 0. "
                             + "If the prompt is 'shoes' without laces, but the shoes have laces this is somewhat aligned and gets a score of 1. "
                             + "If the prompt is 'a concert without fans', but there's fans in the image, pick the images that show less fans."
                             + "\n\n")
    system_prompt_quality = ("You are a photoshop expert judging which set of images is better composition quality."
                             +"\n\n**Score**: Compositional quality scores can be 2 (very high quality), 1 (visually aesthetic but has elements with distortion/missing features/extra features), 0 (low visual quality, issues with texture/blur/visual artifacts). "
                             +"\nComposition can be broken down into three main aspects 1) Clarity, 2) Disfigured Parts and 3) Movement: \n - Clarity: If the image is blurry, poorly lit, or has poor composition (objects obstructing each other), it gets scores 0. "
                             +"\n - Disfigured Parts: This applies to both body parts of humans and animals as well as objects like motorcycles. If the image has a hand that has 6 fingers it gets a 1 for having otherwise normal fingers, but the hand should not have two fingers. If the fingers themselves are disfigured showing lips and teeth warped in, it gets a 0."
                             + "An image with a dog that has two head and no tail gets a score of 0 because it's impossible. A car with deflated wheels get a score of 1. If it has no wheels, it gets a score of 0."
                             + "\n - Movement:  If the sail of a sailboat's sail shows dynamic ripples and ornate patterns, this shows detail and should get a score of 2. If it's monochrome and flat, it gets a score of 1. If it looks like a cartoon and is inconsistent with the environment, give a score of 0. "
                             + "\n\n"
                             )
    if task == "diversity":
        system_prompt = system_prompt_diversity 
    elif task == "alignment":
        system_prompt = system_prompt_alignment
    elif task == "quality":
        system_prompt = system_prompt_quality
    group_vs_image = "Group" if task == "diversity" else "Image"

    payload = {
        "model": "gpt-4-vision-preview",
        "temperature":0.5,
        "messages": [
            {
            "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_per_image( "A", group_vs_image=group_vs_image)},
                ]
                + [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}} for base64_image in base64_image_A]
            }, 

        ],
        "max_tokens": 500
    }



    payload["messages"].append({"role": "user", "content": "ACK"})
    payload["messages"].append( {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_per_image( "B", group_vs_image=group_vs_image)},
                ]
                + [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}} for base64_image in base64_image_B]
            })
        
    payload["messages"].append({"role": "user", "content": "ACK"})
    payload["messages"].append( {
                "role": "user",
                "content": 
                    prompt_by_group(task =task, theme=base_prompt.replace("_", " "))
            })

    response = query_gpt4v(payload)
    return {"result": post_process(response, task=task, group_vs_image=group_vs_image), "prompt": base_prompt}

def main(args):
    lora_is_A = args.lora_is_A
    lora_string = 'A' if lora_is_A else 'B'
    full_log = f"./gpt4v_lora_{lora_string}_{args.task}.json"
    csv_file = f"./gpt4v_lora_{lora_string}_{args.task}.csv"
    path = Path(args.input_path)
    prompts = [ dir.name for dir in path.iterdir()]
    all_completed = []
    if Path(csv_file).exists():
        with open (csv_file, "r") as f:
            for line in f:
                name = line.split(",")[0].strip().lower()
                all_completed.append(name)
                    
    all_completed = list(set(all_completed))
    # filter prompts
    prompts = list(filter(lambda x : prompt_to_filename(x).lower() not in all_completed , prompts))
    print ("Remaining:", len(prompts))

    for base_prompt in prompts[:args.num]:
        if not (Path(path) / base_prompt / "normal").exists() or not (Path(path) / base_prompt / "lora").exists():
            continue
            
        print (base_prompt)
        for _ in range(args.retries):
            response = judge(path, base_prompt, lora_is_A, task = args.task, debug =args.debug)
            result = response["result"]
                
            if len([r for r in result if "Preference" == r["Name"]]) == 1:
                break
            else:
                print ("retrying")
        else:
            continue

        with open (full_log, "a") as f:
            json.dump(response ,f)
        result = response["result"]
        print (result)
        pref = [r for r in result if "Preference"  == r["Name"] ][0]["Choice"]
        A = [r for r in result if r["Name"] == "Group Diversity" and r["Group"]=="A"]
        B = [r for r in result if r["Name"] == "Group Diversity" and r["Group"]=="B"]
        with open (csv_file, "a") as f:
            f.write(f"{base_prompt}, {pref}, {A}, {B}\n")   
            


if __name__=="__main__": 
    parser = argparse.ArgumentParser(description='Evaluate results using GPT4V.')
    parser.add_argument('--lora_is_A', default=True, help='lora is A or B')
    parser.add_argument('--task', required=True, help='eval task: diversity, alignment, quality')
    parser.add_argument('--debug', default=False, help='Debug mode save images to debug folder')
    parser.add_argument('--num', '-n', type=int, default=None, help='maximum number of prompts to test')
    parser.add_argument('--input_path', type=str, required=True, help='input path with images')
    parser.add_argument('--retries', default = 4, help='number of retry, if response is invalid')
    args = parser.parse_args()
    main(args)
    

    
    
    