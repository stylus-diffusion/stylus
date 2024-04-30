from stylus.utils.filename import prompt_to_filename
import argparse
import io
import pandas as pd

def string_preference_only(file):
    string_output = ""
    for line in open(file):
        base_prompt, pref = line.split(",")[0], line.split(",")[1]
        string_output = string_output + f" {base_prompt}, {pref}\n"
    return string_output
    
def main(args):
    experiment = args.experiment
    if experiment == "alignment":
        experiment_tag = "_alignment"
    elif experiment == "quality":
        experiment_tag = "_quality"
    elif experiment == "diversity":
        experiment_tag = ""
        

    file_B = f"./gpt4v_lora_B{experiment_tag}.csv"
    file_A = f"./gpt4v_lora_A{experiment_tag}.csv"
    prompt_file = "./datasets/parti_prompts.csv"

    string_A = string_preference_only(file_A)
    string_B = string_preference_only(file_B)


    data_A = pd.read_csv(io.StringIO(string_A), names=['prompt', 'pref_A'])
    data_B = pd.read_csv(io.StringIO(string_B), names=['prompt', 'pref_B'])


    data_class = pd.read_csv(prompt_file)[['Category',"Challenge", 'caption']]
    data_class["prompt"] = data_class["caption"].apply(lambda x: prompt_to_filename(x).lower())

    merged_data = pd.merge(data_A, data_B, on='prompt', how="inner")
    merged_data['prompt'] = merged_data['prompt'].apply(lambda x: x.strip())

    if args.verbose: 
        print([prompt for prompt in list(data_B['prompt']) if prompt not in list(data_A['prompt'])])
        print([prompt for prompt in list(data_A['prompt']) if prompt not in list(data_B['prompt'])])


    lora_win = merged_data[merged_data['pref_A'].str.strip() == 'A'][merged_data['pref_B'].str.strip() == 'B']
    normal_win = merged_data[merged_data['pref_A'].str.strip() == 'B'][merged_data['pref_B'].str.strip() == 'A']
    tie = pd.concat([merged_data[merged_data['pref_A'].str.strip() == 'B'][ merged_data['pref_B'].str.strip() == 'B'],merged_data[merged_data['pref_A'].str.strip() == 'A' ][ merged_data['pref_B'].str.strip() == 'A']])

    print(pd.merge(lora_win, data_class, on='prompt', how="inner"))
    print ("============================")
    print(pd.merge(normal_win, data_class, on='prompt', how="inner"))
    print ("============================")
    print(pd.merge(tie, data_class, on='prompt', how="inner"))

    print ("success : ", lora_win.shape[0],"\nfail : ", normal_win.shape[0],"\nabstain : ", tie.shape[0])
    print ("total : ", lora_win.shape[0] + normal_win.shape[0] + tie.shape[0])


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Evaluate results using GPT4V.')
    parser.add_argument('--experiment', type=str, default="alignment", help="experiment to run: options are alignment, quality, diversity")
    parser.add_argument('--verbose', type=bool, default=False, help="verbose output")
    args = parser.parse_args()
    main(args)