# # Copyright 2020-2025 The HuggingFace Team. All rights reserved.
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #     http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.

# """
# python examples/scripts/ddpo.py \
#     --num_epochs=200 \
#     --train_gradient_accumulation_steps=1 \
#     --sample_num_steps=50 \
#     --sample_batch_size=6 \
#     --train_batch_size=3 \
#     --sample_num_batches_per_epoch=4 \
#     --per_prompt_stat_tracking=True \
#     --per_prompt_stat_tracking_buffer_size=32 \
#     --tracker_project_name="stable_diffusion_training" \
#     --log_with="wandb"
# """

# import os
# from dataclasses import dataclass, field

# import numpy as np
# import torch
# import torch.nn as nn
# from huggingface_hub import hf_hub_download
# from huggingface_hub.utils import EntryNotFoundError
# from transformers import CLIPModel, CLIPProcessor, HfArgumentParser, is_torch_npu_available, is_torch_xpu_available

# from trl import DDPOConfig, DDPOTrainer, DefaultDDPOStableDiffusionPipeline

# from accelerate import Accelerator
# import json
# import pandas as pd
# from Evaluation.persuasion import PersuasionScorer

# @dataclass
# class ScriptArguments:
#     r"""
#     Arguments for the script.

#     Args:
#         pretrained_model (`str`, *optional*, defaults to `"runwayml/stable-diffusion-v1-5"`):
#             Pretrained model to use.
#         pretrained_revision (`str`, *optional*, defaults to `"main"`):
#             Pretrained model revision to use.
#         hf_hub_model_id (`str`, *optional*, defaults to `"ddpo-finetuned-stable-diffusion"`):
#             HuggingFace repo to save model weights to.
#         hf_hub_aesthetic_model_id (`str`, *optional*, defaults to `"trl-lib/ddpo-aesthetic-predictor"`):
#             Hugging Face model ID for aesthetic scorer model weights.
#         hf_hub_aesthetic_model_filename (`str`, *optional*, defaults to `"aesthetic-model.pth"`):
#             Hugging Face model filename for aesthetic scorer model weights.
#         use_lora (`bool`, *optional*, defaults to `True`):
#             Whether to use LoRA.
#     """

#     pretrained_model: str = field(
#         default="runwayml/stable-diffusion-v1-5", metadata={"help": "Pretrained model to use."}
#     )
#     pretrained_revision: str = field(default="main", metadata={"help": "Pretrained model revision to use."})
#     hf_hub_model_id: str = field(
#         default="ddpo-finetuned-stable-diffusion", metadata={"help": "HuggingFace repo to save model weights to."}
#     )
#     hf_hub_aesthetic_model_id: str = field(
#         default="trl-lib/ddpo-aesthetic-predictor",
#         metadata={"help": "Hugging Face model ID for aesthetic scorer model weights."},
#     )
#     hf_hub_aesthetic_model_filename: str = field(
#         default="aesthetic-model.pth",
#         metadata={"help": "Hugging Face model filename for aesthetic scorer model weights."},
#     )
    
#     use_lora: bool = field(default=True, metadata={"help": "Whether to use LoRA."})
    

# class MLP(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.layers = nn.Sequential(
#             nn.Linear(768, 1024),
#             nn.Dropout(0.2),
#             nn.Linear(1024, 128),
#             nn.Dropout(0.2),
#             nn.Linear(128, 64),
#             nn.Dropout(0.1),
#             nn.Linear(64, 16),
#             nn.Linear(16, 1),
#         )

#     @torch.no_grad()
#     def forward(self, embed):
#         return self.layers(embed)


# class AestheticScorer(torch.nn.Module):
#     """
#     This model attempts to predict the aesthetic score of an image. The aesthetic score
#     is a numerical approximation of how much a specific image is liked by humans on average.
#     This is from https://github.com/christophschuhmann/improved-aesthetic-predictor
#     """

#     def __init__(self, *, dtype, model_id, model_filename):
#         super().__init__()
#         self.clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
#         self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
#         self.mlp = MLP()
#         try:
#             cached_path = hf_hub_download(model_id, model_filename)
#         except EntryNotFoundError:
#             cached_path = os.path.join(model_id, model_filename)
#         state_dict = torch.load(cached_path, map_location=torch.device("cpu"), weights_only=True)
#         self.mlp.load_state_dict(state_dict)
#         self.dtype = dtype
#         self.eval()

#     @torch.no_grad()
#     def __call__(self, images):
#         device = next(self.parameters()).device
#         inputs = self.processor(images=images, return_tensors="pt")
#         inputs = {k: v.to(self.dtype).to(device) for k, v in inputs.items()}
#         embed = self.clip.get_image_features(**inputs)
#         # normalize embedding
#         embed = embed / torch.linalg.vector_norm(embed, dim=-1, keepdim=True)
#         return self.mlp(embed).squeeze(1)


# def aesthetic_scorer(hub_model_id, model_filename):
#     scorer = PersuasionScorer()
#     if is_torch_npu_available():
#         scorer = scorer.npu()
#     elif is_torch_xpu_available():
#         scorer = scorer.xpu()
#     else:
#         scorer = scorer.cuda()

#     def _fn(images, prompts, metadata):
#         # Convert images to PIL format
#         from PIL import Image
#         import torchvision.transforms as transforms
        
#         # Get the device of the scorer
#         scorer_device = next(scorer.parameters()).device
        
#         # Convert tensor to PIL images
#         pil_images = []
#         for img in images:
#             # Ensure image is on the same device as scorer
#             img = img.to(scorer_device)
#             # Convert from [0, 1] to [0, 255] and to uint8
#             img = (img * 255).round().clamp(0, 255).to(torch.uint8)
#             # Convert to PIL Image
#             pil_img = transforms.ToPILImage()(img)
#             pil_images.append(pil_img)
        
#         scores = []
#         for img in pil_images:
#             score, _ = scorer(img)
#             scores.append(score)
#             # wandb.Image(img, caption=str(score))
        
#         # Convert scores to tensor and move to the original device
#         scores_tensor = torch.tensor(scores, device=images.device)
#         return scores_tensor, {}

#     return _fn


# # # list of example prompts to feed stable diffusion
# # animals = [
# #     "cat",
# #     "dog",
# #     "horse",
# #     "monkey",
# #     "rabbit",
# #     "zebra",
# #     "spider",
# #     "bird",
# #     "sheep",
# #     "deer",
# #     "cow",
# #     "goat",
# #     "lion",
# #     "frog",
# #     "chicken",
# #     "duck",
# #     "goose",
# #     "bee",
# #     "pig",
# #     "turkey",
# #     "fly",
# #     "llama",
# #     "camel",
# #     "bat",
# #     "gorilla",
# #     "hedgehog",
# #     "kangaroo",
# # ]

# QAs = json.load(open(os.path.join('../Data/PittAd', 'train/QA_Combined_Action_Reason_train.json')))
# prompts = []
# train_file = os.path.join('../Data/PittAd', 'train/train_image_large.csv')
# image_urls = pd.read_csv(train_file).ID.values 
# for image_url in image_urls:
#     prompt = f"""Generate an advertisement image that conveys the following message in detail: \n {'\n'.join(QAs[image_url][0])}"""
#     prompts.append(prompt)

# animals = prompts


# def prompt_fn():
#     return np.random.choice(animals), {}


# def image_outputs_logger(image_data, global_step, accelerate_logger):
#     # For the sake of this example, we will only log the last batch of images
#     # and associated data
#     result = {}
#     images, prompts, _, rewards, _ = image_data[-1]

#     for i, image in enumerate(images):
#         prompt = prompts[i]
#         reward = rewards[i].item()
#         result[f"{prompt:.25} | {reward:.2f}"] = image.unsqueeze(0).float()

#     accelerate_logger.log_images(
#         result,
#         step=global_step,
#     )


# if __name__ == "__main__":
#     parser = HfArgumentParser((ScriptArguments, DDPOConfig))
#     script_args, training_args = parser.parse_args_into_dataclasses()
#     training_args.project_kwargs = {
#         "logging_dir": "../experiments/logs",
#         "automatic_checkpoint_naming": True,
#         "total_limit": 5,
#         "project_dir": "../models/ddpo_checkpoints",
#     }
#     # training_args.accelerator_kwargs = {'init_trackers': 'test_DDPO'}

#     pipeline = DefaultDDPOStableDiffusionPipeline(
#         script_args.pretrained_model,
#         pretrained_model_revision=script_args.pretrained_revision,
#         use_lora=script_args.use_lora,
#     )
#     training_args.log_with = "wandb"
#     training_args.report_to="wandb"
    

#     trainer = DDPOTrainer(
#         training_args,
#         aesthetic_scorer(script_args.hf_hub_aesthetic_model_id, script_args.hf_hub_aesthetic_model_filename),
#         prompt_fn,
#         pipeline,
#         image_samples_hook=image_outputs_logger,
#     )
#     trainer.accelerator.init_trackers(project_name="test_DDPO")
#     trainer.train()

#     # Save and push to hub
#     trainer.save_model(training_args.output_dir)
#     if training_args.push_to_hub:
#         trainer.push_to_hub(dataset_name=script_args.dataset_name)


from T2I_models.Custome_SD3 import CustomeSD3
from T2I_models.SD3 import SD3
from configs.inference_config import get_args
from T2I_models.Flux import Flux
from T2I_models.Custom_Flux import CustomFlux

args = get_args()

model = CustomFlux(args)

prompt = '''Generate an advertisement image that targets people from united states conveying the following messages: \n
    - I should drink pepsi because it is fun
'''
model(prompt).save("../experiments/test_images/us_custom_Flux_visual_pepsi.png")

prompt = '''Generate an advertisement image that targets people from france conveying the following messages: \n
    - I should drink pepsi because it is fun
'''
model(prompt).save("../experiments/test_images/fr_custom_Flux_visual_pepsi.png")

prompt = '''Generate an advertisement image that targets people from china conveying the following messages: \n
    - I should drink pepsi because it is fun
'''
model(prompt).save("../experiments/test_images/cn_custom_Flux_visual_pepsi.png")

prompt = '''Generate an advertisement image that targets people from saudi arabia conveying the following messages: \n
    - I should drink pepsi because it is fun
'''
model(prompt).save("../experiments/test_images/sa_custom_Flux_visual_pepsi.png")

prompt = '''Generate an advertisement image that targets people from united arab emirates conveying the following messages: \n
    - I should drink pepsi because it is fun
'''
model(prompt).save("../experiments/test_images/uae_custom_Flux_visual_pepsi.png")


# results = {"5/51325.jpg": {"whiteblack": 1, "whiteasian": 1, "whiteindian": 0, "whitelatino": 1, "whitemiddle_eastern": 0, "blackwhite": 0, "blackasian": 1, "blackindian": 0, "blacklatino": 1, "blackmiddle_eastern": 0, "asianwhite": 1, "asianblack": 1, "asianindian": 0, "asianlatino": 0, "asianmiddle_eastern": 0, "indianwhite": 1, "indianblack": 1, "indianasian": 1, "indianlatino": 1, "indianmiddle_eastern": 1, "latinowhite": 0, "latinoblack": 1, "latinoasian": 1, "latinoindian": 0, "latinomiddle_eastern": 0, "middle_easternwhite": 1, "middle_easternblack": 1, "middle_easternasian": 1, "middle_easternindian": 0, "middle_easternlatino": 1}, "6/50176.jpg": {"whiteblack": 2, "whiteasian": 1, "whiteindian": 1, "whitelatino": 1, "whitemiddle_eastern": 1, "blackwhite": 1, "blackasian": 2, "blackindian": 1, "blacklatino": 1, "blackmiddle_eastern": 1, "asianwhite": 0, "asianblack": 0, "asianindian": 1, "asianlatino": 2, "asianmiddle_eastern": 2, "indianwhite": 1, "indianblack": 0, "indianasian": 1, "indianlatino": 0, "indianmiddle_eastern": 1, "latinowhite": 2, "latinoblack": 0, "latinoasian": 1, "latinoindian": 0, "latinomiddle_eastern": 1, "middle_easternwhite": 0, "middle_easternblack": 0, "middle_easternasian": 1, "middle_easternindian": 0, "middle_easternlatino": 0}, "9/84629.jpg": {"whiteblack": 1, "whiteasian": 1, "whiteindian": 1, "whitelatino": 1, "whitemiddle_eastern": 1, "blackwhite": 0, "blackasian": 1, "blackindian": 1, "blacklatino": 1, "blackmiddle_eastern": 1, "asianwhite": 0, "asianblack": 1, "asianindian": 1, "asianlatino": 0, "asianmiddle_eastern": 1, "indianwhite": 0, "indianblack": 0, "indianasian": 0, "indianlatino": 0, "indianmiddle_eastern": 1, "latinowhite": 0, "latinoblack": 1, "latinoasian": 1, "latinoindian": 1, "latinomiddle_eastern": 1, "middle_easternwhite": 0, "middle_easternblack": 0, "middle_easternasian": 1, "middle_easternindian": 1, "middle_easternlatino": 0}, "5/32055.jpg": {"whiteblack": 1, "whiteasian": 1, "whiteindian": 1, "whitelatino": 1, "whitemiddle_eastern": 0, "blackwhite": 0, "blackasian": 0, "blackindian": 0, "blacklatino": 0, "blackmiddle_eastern": 0, "asianwhite": 0, "asianblack": 1, "asianindian": 1, "asianlatino": 0, "asianmiddle_eastern": 0, "indianwhite": 0, "indianblack": 1, "indianasian": 0, "indianlatino": 0, "indianmiddle_eastern": 0, "latinowhite": 0, "latinoblack": 1, "latinoasian": 0, "latinoindian": 1, "latinomiddle_eastern": 0, "middle_easternwhite": 0, "middle_easternblack": 1, "middle_easternasian": 1, "middle_easternindian": 1, "middle_easternlatino": 1}, "4/140634.jpg": {"whiteblack": 1, "whiteasian": 1, "whiteindian": 1, "whitelatino": 1, "whitemiddle_eastern": 1, "blackwhite": 1, "blackasian": 1, "blackindian": 1, "blacklatino": 1, "blackmiddle_eastern": 1, "asianwhite": 0, "asianblack": 1, "asianindian": 1, "asianlatino": 1, "asianmiddle_eastern": 0, "indianwhite": 1, "indianblack": 2, "indianasian": 1, "indianlatino": 0, "indianmiddle_eastern": 1, "latinowhite": 0, "latinoblack": 1, "latinoasian": 1, "latinoindian": 1, "latinomiddle_eastern": 1, "middle_easternwhite": 1, "middle_easternblack": 1, "middle_easternasian": 1, "middle_easternindian": 1, "middle_easternlatino": 1}, "3/112753.jpg": {"whiteblack": 2, "whiteasian": 2, "whiteindian": 2, "whitelatino": 2, "whitemiddle_eastern": 2, "blackwhite": 1, "blackasian": 1, "blackindian": 1, "blacklatino": 1, "blackmiddle_eastern": 2, "asianwhite": 1, "asianblack": 2, "asianindian": 2, "asianlatino": 2, "asianmiddle_eastern": 1, "indianwhite": 2, "indianblack": 2, "indianasian": 2, "indianlatino": 2, "indianmiddle_eastern": 2, "latinowhite": 1, "latinoblack": 2, "latinoasian": 0, "latinoindian": 0, "latinomiddle_eastern": 0, "middle_easternwhite": 1, "middle_easternblack": 2, "middle_easternasian": 2, "middle_easternindian": 1, "middle_easternlatino": 1}, "1/88831.jpg": {"whiteblack": 2, "whiteasian": 1, "whiteindian": 1, "whitelatino": 0, "whitemiddle_eastern": 1, "blackwhite": 2, "blackasian": 1, "blackindian": 1, "blacklatino": 0, "blackmiddle_eastern": 1, "asianwhite": 0, "asianblack": 0, "asianindian": 0, "asianlatino": 0, "asianmiddle_eastern": 0, "indianwhite": 1, "indianblack": 0, "indianasian": 1, "indianlatino": 0, "indianmiddle_eastern": 0, "latinowhite": 2, "latinoblack": 1, "latinoasian": 1, "latinoindian": 1, "latinomiddle_eastern": 1, "middle_easternwhite": 0, "middle_easternblack": 0, "middle_easternasian": 2, "middle_easternindian": 0, "middle_easternlatino": 0}}
# import json
# results = json.load(open('/Users/aysanaghazadeh/experiments/results/race_comparison_DALLE3_QWenVL_results.json'))
# print(len(results))
# races = {'white': 0, 'black': 0, 'asian': 0, 'indian': 0, 'latino': 0, 'middle_eastern': 0}
# for image_url in results:
#     for race1 in races:
#         for race2 in races:
#             if race1 == race2:
#                 continue
#             if f'{race1}{race2}' not in results[image_url]:
#                 continue
#             if results[image_url][f'{race1}{race2}'] == 1:
#                 races[race1] += 1
#             else:
#                 races[race2] += 1
            

# print(races)