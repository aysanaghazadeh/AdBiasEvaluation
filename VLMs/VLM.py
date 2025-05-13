from torch import nn
from VLMs.InternVL2 import InternVL
from VLMs.LLAVA16 import LLAVA16
from VLMs.QWenVL import QWenVL
from VLMs.GPT4_o import GPT4_o
from VLMs.InternVL2_5 import InternVL2_5

class VLM(nn.Module):
    def __init__(self, args):
        super(VLM, self).__init__()
        model_map = {
            'QWenVL': QWenVL,
            'LLAVA16': LLAVA16,
            'InternVL': InternVL,
            'GPT4_o': GPT4_o,
            'InternVL2_5': InternVL2_5
        }
        self.model = model_map[args.VLM](args)

    def forward(self, image, prompt, generate_kwargs=None):
        if generate_kwargs is None:
            output = self.model(image, prompt)
        else:
            output = self.model(image, prompt, generate_kwargs)
        return output

