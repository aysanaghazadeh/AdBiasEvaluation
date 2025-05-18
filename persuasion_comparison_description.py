from configs.inference_config import get_args
from LLMs.LLM import LLM
import pandas as pd
import os
from jinja2 import Environment, FileSystemLoader
import json
import random

def compare_persuasion(pipeline, prompt):
    output = pipeline(prompt)
    print(output)
    output = output.lower().split('answer')[-1]
    print('-'* 100)
    if '1' in output:
        return 1
    elif '2' in output:
        return 2
    else:
        return 0

def description_exists_race(descriptions, image_url):
    races = ['white', 'black', 'asian', 'indian', 'latino', 'middle_eastern']
    for race in races:
        if image_url not in descriptions[race]:
            return False
        
    return True

def description_exists_gender(descriptions, image_url):
    races = ['white', 'black', 'asian', 'indian', 'latino', 'middle_eastern']
    for race in races:
        if image_url not in descriptions[race]:
            return False
        if image_url not in descriptions[f'gender_{race}']:
            return False
    return True

def compare_all_images_race(args):
    images = pd.read_csv(os.path.join(args.result_path, 'results', 'AR_DALLE3_20250507_181113.csv')).image_url.values
    descriptions = {}
    for race in races:
        if os.path.exists(f'../experiments/results/{race}_descriptions.json'):
            descriptions[race] = json.load(open(f'../experiments/results/{race}_descriptions.json'))
        else:
            descriptions[race] = {}
    if os.path.exists(os.path.join(args.result_path, 'results', f'race_description_comparison_DALLE3_{args.LLM}_results.json')):
        image_results = json.load(open(os.path.join(args.result_path, 'results', f'race_desciprtion_comparison_DALLE3_{args.LLM}_results.json')))
    else:
        image_results = {}
    pipeline = LLM(args)
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
        if description_exists_race(descriptions, image_url):
            
            for race1 in races:
                for race2 in races:
                    if race1 == race2:
                        continue
                    
                    description1 = descriptions[race1][image_url]
                    description2 = descriptions[race2][image_url]
                    env = Environment(loader=FileSystemLoader(args.prompt_path))
                    template = env.get_template(args.LLM_prompt)
                    data = {'description1': description1,
                            'description2': description2}
                    prompt = template.render(**data)
                    comparison = compare_persuasion(pipeline, prompt)
                    if comparison == 1:
                        comparisons_win[f'{race1}{race2}'] += 1
                    elif comparison == 2:
                        comparisons_win[f'{race2}{race1}'] += 1
                    print(f'{race1}-{race2} for image {image_url}: {comparison}')
                    image_results[image_url][f'{race1}{race2}'] = 2 - comparison
            json.dump(image_results, open(os.path.join(args.result_path, 'results', f'race_description_comparison_DALLE3_{args.LLM}_results.json'), 'w'))
                    
    print(comparisons_win)
    
def compare_all_images_gender(args):
    images = pd.read_csv(os.path.join(args.result_path, 'results', 'AR_DALLE3_20250507_181113.csv')).image_url.values
    descriptions = {}
    for race in races:
        if os.path.exists(f'../experiments/results/gender_{race}_descriptions.json'):
            descriptions[race] = json.load(open(f'../experiments/results/gender_{race}_descriptions.json'))
        else:
            descriptions[race] = {}
    for race in races:
        if os.path.exists(f'../experiments/results/{race}_descriptions.json'):
            descriptions[race] = json.load(open(f'../experiments/results/{race}_descriptions.json'))
        else:
            descriptions[race] = {}
    if os.path.exists(os.path.join(args.result_path, 'results', f'gender_description_comparison_DALLE3_{args.LLM}_results.json')):
        image_results = json.load(open(os.path.join(args.result_path, 'results', f'gender_description_comparison_DALLE3_{args.LLM}_results.json')))
        print(len(image_results))
    else:
        image_results = {}
    
    pipeline = LLM(args)
    races = ['white', 'black', 'asian', 'indian', 'latino', 'middle_eastern']
    comparisons_win = {'man': 0, 'woman': 0}
    
    for image_url in images:
        if image_url in image_results:
            continue
        image_results[image_url] = {}
        for race in races:
            if description_exists_gender(descriptions, image_url, race):
                env = Environment(loader=FileSystemLoader(args.prompt_path))
                template = env.get_template(args.VLM_prompt)
                description1 = descriptions[race][image_url]
                description2 = descriptions[f'gender_{race}'][image_url]
            
                data = {'options': '1. woman, 2. man',
                        'description1': description1,
                        'description2': description2}
                prompt = template.render(**data)
                comparison = compare_persuasion(pipeline, prompt)
                if comparison == 1:
                    comparisons_win[f'woman'] += 1
                    image_results[image_url]['woman'] = 1
                    image_results[image_url]['man'] = 0
                elif comparison == 2:
                    comparisons_win[f'man'] += 1
                    image_results[image_url]['woman'] = 0
                    image_results[image_url]['man'] = 1
                print(f'woman-man for image {image_url}: {comparison}')
                data = {'options': '1. man, 2. woman',
                        'description1': description1,
                        'description2': description2}
                prompt = template.render(**data)
                comparison = compare_persuasion(pipeline, prompt)
                comparison = compare_persuasion(pipeline, prompt)
                if comparison == 1:
                    comparisons_win[f'man'] += 1
                    if 'man' in image_results[image_url]:
                        image_results[image_url]['man'] += 1
                    else:
                        image_results[image_url]['man'] = 1
                elif comparison == 2:
                    comparisons_win[f'woman'] += 1
                    if 'woman' in image_results[image_url]:
                        image_results[image_url]['woman'] += 1
                    else:
                        image_results[image_url]['woman'] = 1
                    
                print(f'man-woman for image {image_url}: {comparison}')
                
                json.dump(image_results, open(os.path.join(args.result_path, 'results', f'gender_description_comparison_DALLE3_{args.LLM}_results.json'), 'w'))
                    
    print(comparisons_win)
    
                

if __name__ == "__main__":
    args = get_args()
    if args.bias_type == 'race':
        compare_all_images_race(args)
    if args.bias_type == 'gender':
        compare_all_images_gender(args)