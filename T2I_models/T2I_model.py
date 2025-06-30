from torch import nn
from T2I_models.PitxArt import PixArt
from T2I_models.SDXLFlash import SDXL
from T2I_models.DMD2 import DMD
from T2I_models.AuraFlow import AuraFlow
from T2I_models.DALLE3 import DALLE3
from T2I_models.Flux import Flux
from T2I_models.gpt_image import GPT_Image
from T2I_models.SD3 import SD3
from T2I_models.Custome_SD3 import CustomeSD3
from T2I_models.Custome_SD3_early import CustomeSD3Early
from T2I_models.Custome_SD3_late import CustomeSD3Late
from T2I_models.Custome_SD3_no_cultural_component import CustomeSD3NoCulturalComponent
from T2I_models.Custome_SD3_no_image import CustomeSD3NoImage
from T2I_models.Custome_SD3_multiple_images import CustomeSD3MultipleImages

class T2IModel(nn.Module):
    def __init__(self, args):
        super(T2IModel, self).__init__()
        model_map = {
            'PixArt': PixArt,
            'SDXL': SDXL,
            'DMD': DMD,
            'AuraFlow': AuraFlow,
            'DALLE3': DALLE3,
            'Flux': Flux,
            'GPTImage': GPT_Image,
            'SD3': SD3,
            'Custome_SD3': CustomeSD3,
            'Custome_SD3_early': CustomeSD3Early,
            'Custome_SD3_late': CustomeSD3Late,
            'Custome_SD3_ncc': CustomeSD3NoCulturalComponent,
            'Custome_SD3_ni': CustomeSD3NoImage,
            'Custome_SD3_mi': CustomeSD3MultipleImages
        }
        self.model = model_map[args.T2I_model](args)

    def forward(self, prompt):
        return self.model(prompt)
