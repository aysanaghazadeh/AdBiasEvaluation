

from T2I_models.Custome_SD3 import CustomeSD3
from T2I_models.SD3 import SD3
from configs.inference_config import get_args
from T2I_models.Flux import Flux
from T2I_models.T2I_model import T2IModel
import json
import os



if __name__ == "__main__":
    args = get_args()

    model = T2IModel(args)
    ARs = json.load(open('util/data/AR_statements.json'))

    countries = ['france', 'china', 'united arab emirates', 'south africa', 'mexico']
    country_short = ['fr', 'cn', 'uae', 'sa', 'm']
    for i, AR in ARs.items():
        for j, country in enumerate(countries):
            ct = country_short[j]
            print('prompt:', AR)
            print(country)
            prompt = f'''Generate an advertisement image that targets people from {country} conveying the following messages: \n
                - {AR}
            '''
            os.makedirs(f'../experiments/generated_images/sample100/{args.T2I_model}/{ct}', exist_ok=True)
            model(prompt).save(f"../experiments/generated_images/sample100/{args.T2I_model}/{ct}/{i}.png")
