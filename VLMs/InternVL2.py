from torch import nn
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel, CLIPImageProcessor
import torch
import torchvision.transforms as T
from PIL import Image


class InternVL(nn.Module):
    def __init__(self, args):
        super(InternVL, self).__init__()
        self.args = args
        self.model = AutoModel.from_pretrained(
            "OpenGVLab/InternVL2_5-38B",
            # torch_dtype=torch.bfloat16,
            load_in_8bit=True,
            torch_dtype=torch.bfloat16,
            device_map='auto',
            low_cpu_mem_usage=True,
            trust_remote_code=True).eval()
        self.tokenizer = AutoTokenizer.from_pretrained("OpenGVLab/InternVL2_5-38B",
                                                       trust_remote_code=True)
        # path = "OpenGVLab/InternVL-Chat-V1-1"
        # self.model = AutoModel.from_pretrained(
        #     path,
        #     torch_dtype=torch.bfloat16,
        #     low_cpu_mem_usage=True,
        #     use_flash_attn=True,
        #     load_in_8bit=True,
        #     trust_remote_code=True,
        #     device_map='auto').eval()
        # self.tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)

    def build_transform(self, input_size):
        IMAGENET_MEAN = (0.485, 0.456, 0.406)
        IMAGENET_STD = (0.229, 0.224, 0.225)
        MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
        transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD)
        ])
        return transform

    def find_closest_aspect_ratio(self, aspect_ratio, target_ratios, width, height, image_size):
        best_ratio_diff = float('inf')
        best_ratio = (1, 1)
        area = width * height
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        return best_ratio

    def dynamic_preprocess(self, image, min_num=1, max_num=6, image_size=448, use_thumbnail=False):
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height

        # calculate the existing image aspect ratio
        target_ratios = set(
            (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
            i * j <= max_num and i * j >= min_num)
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

        # find the closest aspect ratio to the target
        target_aspect_ratio = self.find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, image_size)

        # calculate the target width and height
        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

        # resize the image
        resized_img = image.resize((target_width, target_height))
        processed_images = []
        for i in range(blocks):
            box = (
                (i % (target_width // image_size)) * image_size,
                (i // (target_width // image_size)) * image_size,
                ((i % (target_width // image_size)) + 1) * image_size,
                ((i // (target_width // image_size)) + 1) * image_size
            )
            # split the image
            split_img = resized_img.crop(box)
            processed_images.append(split_img)
        assert len(processed_images) == blocks
        if use_thumbnail and len(processed_images) != 1:
            thumbnail_img = image.resize((image_size, image_size))
            processed_images.append(thumbnail_img)
        return processed_images

    def load_image(self,image_file, input_size=448, max_num=12):
        image = Image.open(image_file).convert('RGB')
        transform = self.build_transform(input_size=input_size)
        images = self.dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        return pixel_values
    
    def forward(self, images, prompt):
        image_1 = images[0]
        image_2 = images[1]
        generation_config = dict(max_new_tokens=512, do_sample=True)

        # multi-image multi-round conversation, separate images (多图多轮对话，独立图像)
        pixel_values1 = self.load_image(image_1, max_num=12).to(torch.bfloat16).cuda()
        pixel_values2 = self.load_image(image_2, max_num=12).to(torch.bfloat16).cuda()
        pixel_values = torch.cat((pixel_values1, pixel_values2), dim=0)
        num_patches_list = [pixel_values1.size(0), pixel_values2.size(0)]

        question = f'Image-1: <image>\nImage-2: <image>\n{prompt}'
        response, history = self.model.chat(self.tokenizer, pixel_values, question, generation_config,
                                    num_patches_list=num_patches_list,
                                    history=None, return_history=True)
        print(f'User: {question}\nAssistant: {response}')
        return response

        
