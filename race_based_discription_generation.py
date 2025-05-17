import os
from PIL import Image
import pandas as pd
from openai import OpenAI
import base64
import shutil
import json
from configs.inference_config import get_args

def get_image_list(info_file):
    df = pd.read_csv(info_file)
    print(df.head())
    return df.values

if __name__ == "__main__":
    args = get_args()
    client = OpenAI()
    description_file = os.path.join(args.result_path, 
                            f'_race_gender_image_description.json')
    data = json.load(open(description_file))
    races = ["white", "black", "asian", "indian", "latino", "middle_eastern"]
    descriptions = {}
    for race in races:
        if os.path.exists(f'../experiments/results/{race}_descriptions.json'):
            descriptions[race] = json.load(open(f'../experiments/results/{race}_descriptions.json'))
        else:
            descriptions[race] = {}
    for race in races:
        if os.path.exists(f'../experiments/results/gender_{race}_descriptions.json'):
            descriptions[f'gender_{race}'] = json.load(open(f'../experiments/results/gender_{race}_descriptions.json'))
        else:
            descriptions[f'gender_{race}'] = {}
    for image_url in data:
        print(f'describing image {image_url}')
        try:
            for race in races:
                if image_url in descriptions[race] and image_url in descriptions[f'gender_{race}']:
                    continue
                if race != 'white':
                    prompt = f'''In the following description, replace the 'white person' with '{race} person', changing the race characteristic of the person. Only return the new description without any further explanation.
                    {data[image_url]}
                    '''
                    input = [{
                                "role": "user",
                                "content": [
                                    {"type": "input_text", "text": prompt},
                                ]
                            }]
                    response = client.responses.create(
                        model="gpt-4o-2024-08-06",
                        input=input,
                        temperature=0
                    )
                    descriptions[race][image_url] = response.output_text
                else:
                    descriptions[race][image_url] = data[image_url]
                prompt = f'''In the following description, replace the 'white person' with '{race} person' of the opposite gender, changing the race and gender characteristics of the person. Only return the new description without any further explanation.
                {data[image_url]}
                '''
                input = [{
                            "role": "user",
                            "content": [
                                {"type": "input_text", "text": prompt},
                            ]
                        }]
                response = client.responses.create(
                    model="gpt-4o-2024-08-06",
                    input=input,
                    temperature=0
                )
                descriptions[f'gender_{race}'][image_url] = response.output_text
                with open(f'../experiments/results/{race}_descriptions.json', 'w') as file:
                    json.dump(descriptions[race], file)
                with open(f'../experiments/results/gender_{race}_descriptions.json', 'w') as file:
                    json.dump(descriptions[f'gender_{race}'], file)
        except Exception as e:
            print(f"Error processing image {image_url}")
            print(e)
            continue        
                