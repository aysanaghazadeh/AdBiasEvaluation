from torch import nn
from openai import OpenAI
import os
import requests
from PIL import Image
import io

class GPT4o(nn.Module):
    def __init__(self, args):
        super(GPT4o, self).__init__()
        self.args = args
        self.client = OpenAI()

    def forward(self, prompt):
        completion = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        return completion.choices[0].message.content