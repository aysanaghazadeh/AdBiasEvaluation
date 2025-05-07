from torch import nn
from openai import OpenAI
import os
import requests
from PIL import Image
import io
import base64


class GPT_Image(nn.Module):
    def __init__(self, args):
        super(GPT_Image, self).__init__()
        os.environ["OPENAI_API_KEY"] = args.api_key
        self.client = OpenAI()
        print(args.api_key)

    def forward(self, prompt):
        result = self.client.images.generate(
                            model="gpt-image-1",
                            n=1,
                            prompt=prompt
                        )
        image_base64 = result.data[0].b64_json
        image_bytes = base64.b64decode(image_base64)

        # Open the image with PIL
        image = Image.open(image_bytes)
        return image
