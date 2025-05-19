import torch
from torch import nn
from transformers import BitsAndBytesConfig
import torch
from T2I_models.SD3_modified import CustomStableDiffusionPipeline
from transformers import AutoProcessor, CLIPModel
import json
import os
import random
from PIL import Image
from util.data.mapping import COUNTRY_TO_VISUAL

class ProjectionBlock(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.CLIP_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.CLIP_processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.projection_layer = torch.nn.Linear(512, 4096)
        self.texts_cross_attention = torch.nn.MultiheadAttention(embed_dim=4096, num_heads=8, batch_first=True)
        self.CLIP_model.requires_grad_(True)
        self.cross_attention = torch.nn.MultiheadAttention(embed_dim=4096, num_heads=8, batch_first=True)
        self.projection_layer.requires_grad_(True)
        self.cross_attention.requires_grad_(True)
        self.args = args

    def forward(self, image, encoded_prompt, encoded_reason, encoded_cultural_components, time_step):
        
        if time_step < 10:
            return encoded_prompt
        # encoded_cultural_components = torch.cat([e for e in encoded_cultural_components], dim=0)
        
        cultural_components_reason, _ = self.texts_cross_attention(
                                        query=encoded_reason,              # (1, 154, 4096)
                                        key=encoded_cultural_components,          # (1, 1, 4096)
                                        value=encoded_cultural_components         # (1, 1, 4096)
                                    )

        if time_step < 20:
            # print(encoded_prompt.size(), cultural_components_reason.size())
            # print(torch.cat([encoded_prompt, cultural_components_reason], dim=1).size())
            return torch.cat([encoded_cultural_components, encoded_prompt, encoded_reason], dim=1)
        encoded_prompt = encoded_prompt.to(self.args.device)
        inputs = self.CLIP_processor(images=image, return_tensors="pt").to(self.args.device)
        clip_image_features = self.CLIP_model.get_image_features(**inputs)
        clip_image_features = clip_image_features / clip_image_features.norm(p=2, dim=1, keepdim=True)
        clip_image_features = self.projection_layer(clip_image_features)
        clip_image_features = clip_image_features.unsqueeze(1) 
        # print(clip_image_features.size())
        features, _ = self.cross_attention(
                        query=cultural_components_reason,              # (1, 154, 4096)
                        key=clip_image_features,          # (1, 1, 4096)
                        value=clip_image_features         # (1, 1, 4096)
                    )
        # print(features)
        
        features = torch.cat([features, encoded_cultural_components, encoded_prompt], dim=1)
        features = features.to(self.args.device)
        return features

class CustomeSD3(nn.Module):
    def __init__(self, args):
        super(CustomeSD3, self).__init__()
        self.device = args.device
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=torch.float16
        )
        self.args = args
        self.pipeline = CustomStableDiffusionPipeline.from_pretrained(
            'stabilityai/stable-diffusion-3-medium-diffusers',
            torch_dtype=torch.float32,
            load_in_8bit=True,
        ).to(self.device)
        self.projection_block = ProjectionBlock(args)
        # self.projection_block.load_state_dict(torch.load("../models/SD3_modified/checkpoint-500/projection_block.pt"))
        self.projection_block.to(self.device) 
        self.pipeline.projection_block = self.projection_block
        self.country_image_map = json.load(open(os.path.join(args.data_path, "train/countries_image_map_single.json")))
        self.image_cultural_components_map = json.load(open(os.path.join(args.data_path, "train/components.json")))
        

    def forward(self, prompt):
        country = prompt.split("Generate an advertisement image that targets people from ")[-1].split(" conveying the following messages:")[0]
        style_images = self.country_image_map[country]
        style_images = random.sample(style_images, 3)
        print(country)
        print(style_images)
        if country == 'united states':
            negative_style_image = '0/33540.jpg'
        else:
            negative_style_image = '4/85894.jpg'
        style_image = Image.open(os.path.join(self.args.data_path, "train_images_total", style_images[0]))
        negative_style_image = Image.open(os.path.join(self.args.data_path, "train_images_total", negative_style_image))
        if country in COUNTRY_TO_VISUAL:
            cultural_components = COUNTRY_TO_VISUAL[country]
        else:
            cultural_components = ''
        for image in style_images:
            cultural_components += ' ' + ', '.join(self.image_cultural_components_map[image])
        print(cultural_components)
        generator = torch.Generator(device=self.device).manual_seed(0)
        return self.pipeline(prompt=prompt, style_image=style_image, negative_style_image=negative_style_image, cultural_components=cultural_components, generator=generator).images[0]
    
    
    