import torch
from torch import nn
from LLMs.Mistral7B import Mistral7B
from LLMs.LLAMA3 import LLAMA3
from LLMs.phi import Phi
from LLMs.Mistral7BInstruct import Mistral7BInstruct
from LLMs.vicuna import Vicuna
from LLMs.LLAMA3_instruct import LLAMA3Instruct
from LLMs.InternLM import InternLM
from VLMs.InternVL2 import InternVL
from VLMs.LLAVA16 import LLAVA16
from VLMs.QWenVL import QWenVL
# from T2I_models.T2I_model import T2IModel
import wandb

class PersuasionScorer(nn.Module):
    def __init__(self):
        super(PersuasionScorer, self).__init__()
        # self.args = args
        model_map = {
            'Mistral7B': Mistral7B,
            'LLAMA3': LLAMA3,
            'LLAMA3_instruct': LLAMA3Instruct,
            'phi': Phi,
            'Mistral7BInstruct': Mistral7BInstruct,
            'vicuna': Vicuna,
            'InternLM': InternLM,
            'InternVL': InternVL,
            'LLAVA16': LLAVA16,
            'QWenVL': QWenVL
        }
        self.model = model_map['QWenVL']()
        

    @torch.no_grad()
    def forward(self, image):
        def extract_score(output):
            string_value = output.split('Answer: ')[-1]
            # Extract only the digit from the string
            int_value = int(''.join(filter(str.isdigit, string_value)))
            return int_value
        
        eval_prompt = """
                <image>\n USER:
                Context: If the image convinces the audience to take an action it is considered persuasive.
                Question: Based on the context score the persuasiveness of the image in range of (-5, 5).
                Your output format is only Answer: score form, no other form. Empty is not allowed.
                ASSISTANT:
        """
        if 'QWenVL' in ['LLAVA16', 'QWenVL', 'InternVL']:
            output = self.model(image, eval_prompt)
        else:
            output = self.model(eval_prompt)
        
        image = wandb.Image(image, caption=f"score = {output}")
        return extract_score(output), image
