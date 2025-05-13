from configs.inference_config import get_args
from VLMs.VLM import VLM
import pandas as pd
import jinja2
import os
def compare_persuasion(pipeline, images, prompt):
    output = pipeline(images, prompt)
    output = output.lower().split('answer')[-1]
    print(output)
    if '1' in output:
        return 1
    elif '2' in output:
        return 2
    else:
        return 0

def image_exists(image_dir, image_url):
    image_exits = True
    races = ['white', 'black', 'asian', 'indian', 'latino', 'middle_eastern']
    for race in races:
        if os.path.exists(os.path.join(image_dir, race, image_url)):
            continue
        else:
            image_exists = False
    return image_exists

def compare_all_images(args):
    images = pd.read_csv(os.path.join(args.result_path, 'results', 'AR_DALLE3_20250507_181113.csv'))
    image_dir = '/'.join(images.generated_image_url.values[0].split('/')[:-2])
    print(image_dir)
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
        if image_exists(image_dir, image_url):
            for race1 in races:
                for race2 in races:
                    if race1 == race2:
                        continue
                    prompt = jinja2.Template(args.VLM_prompt).render(race1=race1, race2=race2)
                    image1 = os.path.join(args.image_dir, race1, image_url)
                    image2 = os.path.join(args.image_dir, race2, image_url)
                    comparison = compare_persuasion(pipeline, [image1, image2], prompt)
                    if comparison == 1:
                        comparisons_win[f'{race1}{race2}'] += 1
                    elif comparison == 2:
                        comparisons_win[f'{race2}{race1}'] += 1
                    print(f'{race1}{race2} for image {image_url}: {comparison}')
    print(comparisons_win)
                

if __name__ == "__main__":
    args = get_args()
    compare_all_images(args)