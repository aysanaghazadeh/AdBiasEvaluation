import torch
from torch import nn
from transformers import BitsAndBytesConfig
import torch
from T2I_models.Flux_modified import CustomFluxPipeline
from transformers import AutoProcessor, CLIPModel
import json
import os
import random
from PIL import Image
from util.data.mapping import COUNTRY_TO_VISUAL
import ast

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
        
        encoded_cultural_components = encoded_cultural_components.to(self.args.device)
        encoded_reason = encoded_reason.to(self.args.device)
        encoded_prompt = encoded_prompt.to(self.args.device)
        
        if time_step < 20:
            return torch.cat([encoded_prompt, encoded_cultural_components], dim=1)
        
        inputs = self.CLIP_processor(images=image, return_tensors="pt").to(self.args.device)
        clip_image_features = self.CLIP_model.get_image_features(**inputs)
        clip_image_features = clip_image_features / clip_image_features.norm(p=2, dim=1, keepdim=True)
        clip_image_features = self.projection_layer(clip_image_features)
        clip_image_features = clip_image_features.unsqueeze(1) 
        # print(clip_image_features.size())
        cultural_components_reason, _ = self.texts_cross_attention(
                                        query=encoded_reason,              # (1, 154, 4096)
                                        key=encoded_cultural_components,          # (1, 1, 4096)
                                        value=encoded_cultural_components         # (1, 1, 4096)
                                    )

        features, _ = self.cross_attention(
                        query=cultural_components_reason,              # (1, 154, 4096)
                        key=clip_image_features,          # (1, 1, 4096)
                        value=clip_image_features         # (1, 1, 4096)
                    )
        # print(features)
        
        features = torch.cat([encoded_prompt, encoded_cultural_components, features], dim=1)
        features = features.to(self.args.device)
        return features

class CustomFlux(nn.Module):
    def __init__(self, args):
        super(CustomFlux, self).__init__()
        self.device = args.device
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=torch.float16
        )
        self.args = args
        self.pipeline = CustomFluxPipeline.from_pretrained(
            'black-forest-labs/FLUX.1-dev',
            torch_dtype=torch.float32,
            load_in_8bit=True,
        ).to(self.args.device)
        self.projection_block = ProjectionBlock(args)
        if not args.train and args.fine_tuned:
            self.projection_block.load_state_dict(torch.load(os.path.join(args.model_path, "SD3_modified/checkpoint-500/projection_block.pt")))
        self.projection_block.to(self.args.device) 
        self.pipeline.projection_block = self.projection_block
        self.country_image_map = json.load(open(os.path.join(args.data_path, "train/countries_image_map.json")))
        self.image_cultural_components_map = json.load(open(os.path.join(args.data_path, "train/components.json")))
        self.topics = json.load(open(os.path.join(args.data_path, "train/Topics_train.json")))
        

    def forward(self, prompt, topic=None):
        # if topic is None and "Topic: " in prompt:
        #     topic = ast.literal_eval(prompt.split("Topic: ")[-1].split("Prompt:")[0].strip())[0]
        #     prompt = prompt.split("Prompt:")[1]
        
        country = prompt.split("Generate an advertisement image that targets people from ")[-1].split(" conveying the following messages:")[0]
        visual_element = COUNTRY_TO_VISUAL[country]
        style_images = self.country_image_map[country]
        
        if len(style_images) > 3:
            style_images = random.sample(style_images, 3)
        # same_topic_images = []
        # for image in style_images:
        #     for topic_id in self.topics[image]:
        #         if topic_id in TOPIC_MAP:
        #             image_topic = TOPIC_MAP[topic_id]
        #         else:
        #             image_topic = topic_id
        #         if topic in image_topic:
        #             same_topic_images.append(image)
        #             break
        
        # if len(same_topic_images) < 5:
        #     style_images = random.sample(style_images, 3)
        # else:
        #     style_images = random.sample(same_topic_images, 3)
            
        print(country)
        print(style_images)
        if country == 'united states':
            negative_style_image = '0/33540.jpg'
        else:
            negative_style_image = '4/85894.jpg'
        style_image = Image.open(os.path.join(self.args.data_path, "train_images_total", style_images[0]))
        negative_style_image = Image.open(os.path.join(self.args.data_path, "train_images_total", negative_style_image))
        components = set()
        for image in style_images:
            image_components = self.image_cultural_components_map[image]
            for component in image_components:
                if component[-4:] != 'text':
                    components.add(component)
                
        components = list(components)
        cultural_components = ', '.join(components) 
        
        print(cultural_components)
        generator = torch.Generator(device=self.args.device).manual_seed(0)
        return self.pipeline(prompt=prompt, style_image=style_image, negative_style_image=negative_style_image, cultural_components=cultural_components, country=country, generator=generator).images[0]
    
    
    
