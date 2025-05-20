import ImageReward as RM
import os
from configs.evaluation_config import get_args
import json


if __name__ == "__main__":
    args = get_args()
    
    model = RM.load("ImageReward-v1.0")
    AR_statements = json.load(open('util/data/AR_statements.json'))
    image_directory = os.path.join(args.result_path, 'generated_images/sample100', args.T2I_model)
    images = [f'{i}.png' for i in list(AR_statements.keys())[0:89]]
    images_scores = {}
    # clip_flant5_score = t2v_metrics.VQAScore(model='clip-flant5-xl') # our recommended scoring model
    
    countries = ['france', 'china', 'united arab emirates', 'south africa', 'mexico']
    country_short = ['fr', 'cn', 'uae', 'sa', 'm']

    for filename in images:
        images_scores[filename] = {}
        for i, country in enumerate(countries):
            ct = country_short[i]
            
            image = os.path.join(image_directory, ct, filename)
            text_AR = AR_statements[filename.split('.')[0]]
            image_AR_score = model.score(text_AR, [image])
            image_country_score = model.score(country, [image])
            average_score = (image_AR_score + image_country_score) / 2 
            images_scores[filename][country] = [average_score, image_AR_score, image_country_score]
        print(filename, images_scores[filename])
        with open(os.path.join(args.result_path, 'results', f'Image_Reward_score_{args.T2I_model}.json'), 'w') as file:
            json.dump(images_scores, file)
            

    