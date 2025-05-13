from transformers import pipeline
from torch import nn
import base64
import requests

class InternVL2_5(nn.Module):
    def __init__(self, args):
        super(InternVL2_5, self).__init__()
        self.pipe = pipeline("image-text-to-text", model="OpenGVLab/InternVL2_5-38B", trust_remote_code=True, device_map="auto")

    def forward(self, images, prompt):
        def encode_image(image_path):
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        messages = [{
                    "role": "user",
                    "content": [
                        
                    ]
                }]
        for image in images:
            messages[0]["content"].append({
                "type": "input_image",
                "image_url": f"data:image/jpeg;base64,{encode_image(image)}"
            })
        messages[0]["content"].append({"type": "input_text", "text": prompt})
        response = self.pipe(messages)
        print(response)
        return response
        
        
        
        