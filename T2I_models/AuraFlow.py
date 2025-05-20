import torch
from torch import nn
from diffusers import AuraFlowPipeline, DiffusionPipeline
from transformers import BitsAndBytesConfig


class AuraFlow(nn.Module):
    def __init__(self, args):
        super(AuraFlow, self).__init__()
        self.device = args.device
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=torch.float16
        )
        self.pipeline = DiffusionPipeline.from_pretrained(
                            "/u/aya34/HF_CACHE/hub/models--fal--AuraFlow-v0.2/snapshots/ea13150f559b7f85d2c5959297f7de10325584b4/transformer",
                            torch_dtype=torch.float16,
                            variant="fp16"  # This tells it to look for `*.fp16.safetensors`
                        ).to("cuda")

    def forward(self, prompt):
        image = self.pipeline(
                    prompt=prompt,
                    height=1024,
                    width=1024,
                    num_inference_steps=28,
                    generator=torch.Generator().manual_seed(666),
                    guidance_scale=5,
                    ).images[0]
        return image
