import torch
from torch import nn
from transformers import BitsAndBytesConfig
import torch
from Training.train_modifiedSD3 import ProjectionBlock
from T2I_models.SD3_modified import CustomStableDiffusionPipeline
from transformers import AutoProcessor, CLIPModel
from T2I_models.SD3_modified import CustomStableDiffusionPipeline

class ProjectionBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.CLIP_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.CLIP_processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.projection_layer = torch.nn.Linear(512, 4096)
        self.texts_cross_attention = torch.nn.MultiheadAttention(embed_dim=4096, num_heads=8, batch_first=True)
        self.CLIP_model.requires_grad_(True)
        self.projection_layer.requires_grad_(True)
        self.cross_attention.requires_grad_(True)

    def forward(self, image, encoded_prompt, encoded_reason, encoded_cultural_components):
        encoded_cultural_components = torch.cat([e for e in encoded_cultural_components], dim=0)
        cultural_components_reason = self.texts_cross_attention(
                                        query=encoded_reason,              # (1, 154, 4096)
                                        key=encoded_cultural_components,          # (1, 1, 4096)
                                        value=encoded_cultural_components         # (1, 1, 4096)
                                    )
        
        encoded_prompt_device = encoded_prompt.device
        encoded_prompt = encoded_prompt.to(self.CLIP_model.device)
        inputs = self.CLIP_processor(images=image, return_tensors="pt").to(self.CLIP_model.device)
        clip_image_features = self.CLIP_model.get_image_features(**inputs)
        clip_image_features = clip_image_features / clip_image_features.norm(p=2, dim=1, keepdim=True)
        clip_image_features = self.projection_layer(clip_image_features)
        clip_image_features = clip_image_features.unsqueeze(1) 
        # print(clip_image_features.size())
        features, _ = self.cross_attention(
                        query=encoded_prompt,              # (1, 154, 4096)
                        key=clip_image_features,          # (1, 1, 4096)
                        value=clip_image_features         # (1, 1, 4096)
                    )
        # print(features)
        features = torch.cat([features, encoded_prompt], dim=1)
        features = features.to(encoded_prompt_device)
        return features

class CustomeSD3(nn.Module):
    def __init__(self, args):
        super(CustomeSD3, self).__init__()
        self.device = args.device
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=torch.float16
        )
        self.pipeline = CustomStableDiffusionPipeline.from_pretrained(
            'stabilityai/stable-diffusion-3-medium-diffusers',
            torch_dtype=torch.float32,
            load_in_8bit=True,
        ).to(self.device)
        self.projection_block = ProjectionBlock()
        self.projection_block.load_state_dict(torch.load("SD3_finetuned_projection_only/checkpoint-2000/projection_block.pt"))
        self.projection_block.to(self.device) 
        self.pipeline.projection_block = self.projection_block

    def forward(self, prompt, style_image, negative_style_image):
        generator = torch.Generator(device=self.device).manual_seed(0)
        return self.pipeline(prompt=prompt, style_image=style_image, negative_style_image=negative_style_image, generator=generator).images[0]
    
    
    
