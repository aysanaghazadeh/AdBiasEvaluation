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
        descriptions[race] = {}
    for race in races:
        descriptions[f'gender_{race}'] = {}
    for image_url in data:
        for race in races:
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
            
            