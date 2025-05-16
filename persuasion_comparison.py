from configs.inference_config import get_args
from VLMs.VLM import VLM
import pandas as pd
import os
from jinja2 import Environment, FileSystemLoader
import json
import random

def compare_persuasion(pipeline, images, prompt):
    output = pipeline(images, prompt)
    print(output)
    output = output.lower().split('answer')[-1]
    print('-'* 100)
    if '1' in output:
        return 1
    elif '2' in output:
        return 2
    else:
        return 0

def image_exists_race(image_dir, image_url):
    image_exists = True
    races = ['white', 'black', 'asian', 'indian', 'latino', 'middle_eastern']
    for race in races:
        if not os.path.exists(os.path.join(image_dir, race, image_url)):
            print(os.path.join(image_dir, race, image_url))
            image_exists = False
    return image_exists

def image_exists_gender(image_dir, image_url, race):
    if not os.path.exists(os.path.join(image_dir, f'gender_{race}', image_url)) or not os.path.exists(os.path.join(image_dir, race, image_url)):
        print(os.path.join(image_dir, race, image_url))
        return False
    return True

def compare_all_images_race(args):
    images = pd.read_csv(os.path.join(args.result_path, 'results', 'AR_DALLE3_20250507_181113.csv'))
    image_dir = '/'.join(images.generated_image_url.values[0].split('/')[:-3])
    if os.path.exists(os.path.join(args.result_path, 'results', f'race_comparison_DALLE3_{args.VLM}_results.json')):
        image_results = json.load(open(os.path.join(args.result_path, 'results', f'race_comparison_DALLE3_{args.VLM}_results.json')))
    else:
        image_results = {}
    images = images.image_url.values
    pipeline = VLM(args)
    races = ['white', 'black', 'asian', 'indian', 'latino', 'middle_eastern']
    comparisons_win = {}
    for race1 in races:
        for race2 in races:
            if race1 == race2:
                continue
            comparisons_win[f'{race1}{race2}'] = 0
    for image_url in images:
        if image_url in image_results:
            continue
        image_results[image_url] = {}
        if image_exists_race(image_dir, image_url):
            
            for race1 in races:
                for race2 in races:
                    if race1 == race2:
                        continue
                    env = Environment(loader=FileSystemLoader(args.prompt_path))
                    template = env.get_template(args.VLM_prompt)
                    prompt = template.render()
                    image1 = os.path.join(image_dir, race1, image_url)
                    image2 = os.path.join(image_dir, race2, image_url)
                    comparison = compare_persuasion(pipeline, [image1, image2], prompt)
                    if comparison == 1:
                        comparisons_win[f'{race1}{race2}'] += 1
                    elif comparison == 2:
                        comparisons_win[f'{race2}{race1}'] += 1
                    print(f'{race1}{race2} for image {image_url}: {comparison}')
                    image_results[image_url][f'{race1}{race2}'] = 2 - comparison
            json.dump(image_results, open(os.path.join(args.result_path, 'results', f'race_comparison_DALLE3_{args.VLM}_results.json'), 'w'))
                    
    print(comparisons_win)
    
def compare_all_images_gender(args):
    images = pd.read_csv(os.path.join(args.result_path, 'results', 'AR_DALLE3_20250507_181113.csv'))
    image_dir = '/'.join(images.generated_image_url.values[0].split('/')[:-3])
    if os.path.exists(os.path.join(args.result_path, 'results', f'gender_comparison_DALLE3_{args.VLM}_results.json')):
        image_results = json.load(open(os.path.join(args.result_path, 'results', f'gender_comparison_DALLE3_{args.VLM}_results.json')))
        print(len(image_results))
    else:
        image_results = {}
    images = images.image_url.values
    pipeline = VLM(args)
    races = ['black', 'asian', 'indian', 'latino', 'middle_eastern']
    comparisons_win = {'man': 0, 'woman': 0}
    
    for image_url in images:
        if image_url in image_results:
            continue
        image_results[image_url] = {}
        race = random.choice(races)
        if image_exists_gender(image_dir, image_url, race):
            env = Environment(loader=FileSystemLoader(args.prompt_path))
            template = env.get_template(args.VLM_prompt)
            
            image1 = os.path.join(image_dir, race, image_url)
            image2 = os.path.join(image_dir, f'gender_{race}', image_url)
        
            data = {'options': '1. woman, 2. man'}
            prompt = template.render(**data)
            comparison = compare_persuasion(pipeline, [image1, image2], prompt)
            if comparison == 1:
                comparisons_win[f'woman'] += 1
            elif comparison == 2:
                comparisons_win[f'man'] += 1
            print(f'woman-man for image {image_url}: {comparison}')
            data = {'options': '1. man, 2. woman'}
            prompt = template.render(**data)
            comparison = compare_persuasion(pipeline, [image1, image2], prompt)
            comparison = compare_persuasion(pipeline, [image1, image2], prompt)
            if comparison == 1:
                comparisons_win[f'man'] += 1
            elif comparison == 2:
                comparisons_win[f'woman'] += 1
            print(f'man-woman for image {image_url}: {comparison}')
            
            json.dump(image_results, open(os.path.join(args.result_path, 'results', f'gender_comparison_DALLE3_{args.VLM}_results.json'), 'w'))
                    
    print(comparisons_win)
    
                

if __name__ == "__main__":
    args = get_args()
    if args.bias_type == 'race':
        compare_all_images_race(args)
    if args.bias_type == 'gender':
        compare_all_images_gender(args)