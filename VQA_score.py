import t2v_metrics
import os
from configs.evaluation_config import get_args
import json


if __name__ == "__main__":
    args = get_args()
    clip_flant5_score = t2v_metrics.VQAScore(model='clip-flant5-xxl') # our recommended scoring model
    AR_statements = json.load(open('util/data/AR_statements.json'))
    image_directory = os.path.join(args.result_path, 'generated_images/sample100', args.T2I_model)
    images = [f'{i}.jpg' for i in list(AR_statements.keys)]
    images_scores = {}
    # clip_flant5_score = t2v_metrics.VQAScore(model='clip-flant5-xl') # our recommended scoring model
    
    countries = ['france', 'china', 'united arab emirates', 'south africa', 'mexico']
    country_short = ['fr', 'cn', 'uae', 'sa', 'm']

    for filename in images:
        for i, country in enumerate(countries):
            ct = country_short[i]
            
            image = os.path.join(image_directory, ct, filename)
            text_AR = AR_statements[filename.split('.')[0]]
            image_AR_score = clip_flant5_score(images=[image], texts=[text_AR])
            image_country_score = clip_flant5_score(images=[image], texts=[country])
            average_score = (image_AR_score + image_country_score) / 2 
            images_scores[filename][country] = [average_score, image_AR_score, image_country_score]
    with open(os.path.join(args.result_path, 'results', f'VQA_score_{args.T2I_model}.json'), 'w') as file:
        json.dump(images_scores, file)
            

    