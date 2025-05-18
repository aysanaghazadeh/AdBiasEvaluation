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
            # messages = [
            #     {"role": "user", "content": prompt},
            # ]
            response, history = self.model.chat(self.tokenizer, prompt, history=[])
            print(response)
            return response
            # length = 0
            # for response, history in self.model.stream_chat(self.tokenizer, prompt, history=[]):
            #     output = history[0][-1]
            #     length = len(response)
            # return output
            
            # inputs = self.tokenizer([prompt], return_tensors="pt")
            # for k, v in inputs.items():
            #     inputs[k] = v.cuda()
            # gen_kwargs = {"temperature": 0.8, "do_sample": True}
            # output = self.model.generate(**inputs, **gen_kwargs)
            # output = self.tokenizer.decode(output[0].tolist(), skip_special_tokens=True)
            # return output

