from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import os
from PIL import Image
class QWenVL(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-VL-32B-Instruct", torch_dtype="auto", device_map="auto")
        self.model = self.model.to('cuda')
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-32B-Instruct")


    def forward(self, images, prompt, generate_kwargs={'max_new_tokens': 20}):

        messages = [
            {
                "role": "user",
                "content": [
                    
                   
                ],
            }
        ]
        for image in images:
            messages[0]["content"].append({
                "type": "image",
                "image": Image.open(image),
            })
        messages[0]["content"].append({
            "type": "text",
            "text": prompt,
        })

        # Preparation for inference
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to('cuda')

        # Inference: Generation of the output
        generated_ids = self.model.generate(**inputs,
                                            max_new_tokens=generate_kwargs['max_new_tokens'])
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        print(output_text)
        return output_text[0]
