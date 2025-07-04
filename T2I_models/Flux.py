from diffusers import FluxPipeline, DiffusionPipeline
import torch
from torch import nn
from transformers import BitsAndBytesConfig


class Flux(nn.Module):
    def __init__(self, args):
        super(Flux, self).__init__()
        self.args = args
        self.device = args.device
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=torch.float16
        )
        self.pipeline = DiffusionPipeline.from_pretrained("black-forest-labs/FLUX.1-dev",
                                                           torch_dtype=torch.float16)
                                                        #   quantization_config=quantization_config)
        self.pipeline = self.pipeline.to(device=args.device)

    def forward(self, prompt):
        generator = torch.Generator(device=self.args.device).manual_seed(0)
        image = self.pipeline(prompt, generator=generator).images[0]
        return image
