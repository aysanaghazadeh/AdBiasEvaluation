# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
python examples/scripts/ddpo.py \
    --num_epochs=200 \
    --train_gradient_accumulation_steps=1 \
    --sample_num_steps=50 \
    --sample_batch_size=6 \
    --train_batch_size=3 \
    --sample_num_batches_per_epoch=4 \
    --per_prompt_stat_tracking=True \
    --per_prompt_stat_tracking_buffer_size=32 \
    --tracker_project_name="stable_diffusion_training" \
    --log_with="wandb"
"""

import os
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import EntryNotFoundError
from transformers import CLIPModel, CLIPProcessor, HfArgumentParser, is_torch_npu_available, is_torch_xpu_available

from trl import DDPOConfig, DDPOTrainer, DefaultDDPOStableDiffusionPipeline
from Evaluation.persuasion import PersuasionScorer
from util.data.data_util import get_train_DDPO_persuasion_Dataset
from configs.training_config import get_args
import wandb

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(768, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )

    @torch.no_grad()
    def forward(self, embed):
        return self.layers(embed)


class AestheticScorer(torch.nn.Module):
    """
    This model attempts to predict the aesthetic score of an image. The aesthetic score
    is a numerical approximation of how much a specific image is liked by humans on average.
    This is from https://github.com/christophschuhmann/improved-aesthetic-predictor
    """

    def __init__(self, *, dtype, model_id, model_filename):
        super().__init__()
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.mlp = MLP()
        try:
            cached_path = hf_hub_download(model_id, model_filename)
        except EntryNotFoundError:
            cached_path = os.path.join(model_id, model_filename)
        state_dict = torch.load(cached_path, map_location=torch.device("cpu"), weights_only=True)
        self.mlp.load_state_dict(state_dict)
        self.dtype = dtype
        self.eval()

    @torch.no_grad()
    def __call__(self, images):
        device = next(self.parameters()).device
        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {k: v.to(self.dtype).to(device) for k, v in inputs.items()}
        embed = self.clip.get_image_features(**inputs)
        # normalize embedding
        embed = embed / torch.linalg.vector_norm(embed, dim=-1, keepdim=True)
        return self.mlp(embed).squeeze(1)



def persuasion_scorer(args):
    scorer = PersuasionScorer(
        args=args
    )
    if is_torch_npu_available():
        scorer = scorer.npu()
    elif is_torch_xpu_available():
        scorer = scorer.xpu()
    else:
        scorer = scorer.cuda()

    def _fn(images, prompts, metadata):
        # Convert images to PIL format
        from PIL import Image
        import torchvision.transforms as transforms
        
        # Get the device of the scorer
        scorer_device = next(scorer.parameters()).device
        
        # Convert tensor to PIL images
        pil_images = []
        for img in images:
            # Ensure image is on the same device as scorer
            img = img.to(scorer_device)
            # Convert from [0, 1] to [0, 255] and to uint8
            img = (img * 255).round().clamp(0, 255).to(torch.uint8)
            # Convert to PIL Image
            pil_img = transforms.ToPILImage()(img)
            pil_images.append(pil_img)
        
        scores = []
        for img in pil_images:
            score, _ = scorer(img)
            scores.append(score)
            # wandb.Image(img, caption=str(score))
        
        # Convert scores to tensor and move to the original device
        scores_tensor = torch.tensor(scores, device=images.device)
        return scores_tensor, {}

    return _fn





def prompt_fn(animals):
    def _prompt_fn():
        return np.random.choice(animals), {}
    return _prompt_fn


def image_outputs_logger(image_data, global_step, accelerate_logger):
    # For the sake of this example, we will only log the last batch of images
    # and associated data
    result = {}
    images, prompts, _, rewards, _ = image_data[-1]

    for i, image in enumerate(images):
        prompt = prompts[i]
        reward = rewards[i].item()
        result[f"{prompt:.25} | {reward:.2f}"] = image.unsqueeze(0).float()

    accelerate_logger.log_images(
        result,
        step=global_step,
    )


def train(args):
    # list of example prompts to feed stable diffusion
    animals = get_train_DDPO_persuasion_Dataset(args)
    
    # Create output directory if it doesn't exist
    # output_dir = os.path.join(args.result_path, "ddpo_checkpoints")
    # os.makedirs(output_dir, exist_ok=True)
    
    # Create DDPOConfig with our arguments
    training_args = DDPOConfig(
        num_epochs=args.epoch,
        train_gradient_accumulation_steps=4,
        sample_num_steps=10,
        sample_batch_size=args.batch_size,
        train_batch_size=args.batch_size,
        sample_num_batches_per_epoch=4,
        per_prompt_stat_tracking=True,
        per_prompt_stat_tracking_buffer_size=32,
        tracker_project_name="stable_diffusion_training",
        log_with="wandb",
        push_to_hub=False,
        output_dir=args.output_dir
    )

    pipeline = DefaultDDPOStableDiffusionPipeline(
        args.pretrained_model,
        pretrained_model_revision=args.pretrained_revision,
        use_lora=args.use_lora,
    )

    trainer = DDPOTrainer(
        training_args,
        persuasion_scorer(args),
        prompt_fn(animals),
        pipeline,
        image_samples_hook=image_outputs_logger,
    )

    trainer.train()

    # Save and push to hub
    trainer.save_model(args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=args.hf_hub_model_id)