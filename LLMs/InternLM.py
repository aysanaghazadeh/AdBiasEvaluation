from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from peft import PeftModel
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


class InternLM(nn.Module):
    def __init__(self, args):
        super(InternLM, self).__init__()
        self.args = args
        if not args.train:
            
            self.tokenizer = AutoTokenizer.from_pretrained("internlm/internlm2_5-7b-chat", trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained("internlm/internlm2_5-7b-chat", torch_dtype=torch.float16, trust_remote_code=True).cuda()
            self.model = self.model.eval()

    def forward(self, prompt):
        if not self.args.train:
            
            response, history = self.model.chat(self.tokenizer, prompt, history=[])
            return response
            