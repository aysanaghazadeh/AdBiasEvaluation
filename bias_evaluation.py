from jinja2 import Environment, FileSystemLoader
from util.data.trian_test_split import get_test_data, get_train_data
from PIL import Image
import os
import csv
from configs.inference_config import get_args
from util.prompt_engineering.prompt_generation import PromptGenerator
from VLMs.VLM import VLM

def get_model(args):
    model = VLM(args)
    return model

def get_single_charachterestic(args, image, model, charecteristic):
    env = Environment(loader=FileSystemLoader(args.prompt_path))
    template = env.get_template(args.VLM_prompt)
    data = {'characteistic': charecteristic}
    prompt = template.render(**data)
    outputs = model(image, prompt=prompt, generate_kwargs={"max_new_tokens": 250})
    outputs = outputs.split('Answer:')[-1].strip()
    return int('yes' in outputs.lower())

def get_charecteristics(args, image_url, model):
    image_path_psa = os.path.join(args.data_path, args.test_set_images[0], image_url)
    image_path_com = os.path.join(args.data_path, args.test_set_images[1], image_url)
    if os.path.exists(image_path_psa):
        image = Image.open(image_path_psa)
    else:
        image = Image.open(image_path_com)
    charecteristics = {'woman': 0, 'man': 0, 'white': 0, 'black': 0, 'asian': 0, 'indian': 0, 'middle eastern': 0, 'latino': 0, 'native american': 0, 'pacific islander': 0}
    for charecteristic in charecteristics:
        charecteristics[charecteristic] = get_single_charachterestic(args, image, model, charecteristic)
    return charecteristics

def get_charecteristics_for_all_images(args, model):
    images = get_test_data(args)
    charecteristics = {'woman': 0, 'man': 0, 'white': 0, 'black': 0, 'asian': 0, 'indian': 0, 'middle eastern': 0, 'latino': 0, 'native american': 0, 'pacific islander': 0}
    saving_path = os.path.join(args.result_path, 
                               f'charecteristics_{args.test_set_images[0].split("/")[-1].replace(".csv", "")}.csv')
    
    header = ['image_url'] + ['woman', 'man', 'non binary', 'white', 'black', 'asian', 'indian', 'middle eastern', 'latino', 'native american', 'pacific islander']
    with open(saving_path, 'w') as f:
        csv.writer(f).writerow(header)
        print('saving intiated')
    for image_url in images:
        if image_url[0] == 'ID':
            continue
        charecteristic = get_charecteristics(args, image_url[0], model)
        for key in charecteristic:
            charecteristics[key] += charecteristic[key]
        with open(saving_path, 'a') as f:
            csv.writer(f).writerow([image_url[0]] + list(charecteristic.values()))
        print(f'charecteristics for {image_url[0]} are {charecteristic}')
    return charecteristics


if __name__ == "__main__":
    args = get_args()
    model = get_model(args)
    characteristics = get_charecteristics_for_all_images(args, model)
    print(characteristics)
