import torch
from torch import nn
from diffusers import StableDiffusion3Pipeline

class SD3(nn.Module):
    def __init__(self, args):
        super(SD3, self).__init__()
        self.device = args.device
        if not args.train:
            self.pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float16).to(device=args.device)
            

    def forward(self, prompt):
        # negative_prompt = "typical,(deformed, distorted, disfigured:1.3), poorly drawn, bad anatomy, wrong anatomy, " \
        #                   "extra limb, missing limb, floating limbs, (mutated hands and fingers:1.4), disconnected " \
        #                   "limbs, mutation, mutated, ugly, disgusting, blurry, amputation, NSFW "
        generator = torch.Generator(device=self.args.device).manual_seed(42)
        image = self.pipe(
                        prompt,
                        negative_prompt="",
                        num_inference_steps=28,
                        guidance_scale=7.0,
                        generator=generator
                    ).images[0]
        return image
