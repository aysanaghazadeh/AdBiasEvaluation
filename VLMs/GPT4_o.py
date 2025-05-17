from openai import OpenAI
import base64
import requests
from torch import nn

class GPT4_o(nn.Module):
    def __init__(self, args):
        super(GPT4_o, self).__init__()
        self.client = OpenAI()

    def forward(self, images, prompt):
        
        def encode_image(image_path):
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        input = [{
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                    ]
                }]
        if type(images) == list:
            for image in images:
                input[0]["content"].append({
                    "type": "input_image",
                    "image_url": f"data:image/jpeg;base64,{encode_image(image)}"
                })
        else:
            input[0]["content"].append({
                    "type": "input_image",
                    "image_url": f"data:image/jpeg;base64,{encode_image(images)}"
                })
        response = self.client.responses.create(
            model="gpt-4o-2024-08-06",
            input=input,
            temperature=0
        )
        return response.output_text
        
        
        
        