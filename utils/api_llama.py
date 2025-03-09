# Ref: https://github.com/kojima-takeshi188/zero_shot_cot
# Ref: https://github.com/sylinrl/TruthfulQA/blob/main/truthfulqa/metrics.py
# Ref: https://github.com/sylinrl/TruthfulQA/blob/main/truthfulqa/utilities.py

import re, pdb
import os
import json
import random
import transformers
from tqdm import tqdm
import argparse
import pandas as pd

import ssl
import urllib.request
import zipfile



# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

def LlamaChatCompletion(model_name, prompt, max_tokens, model, tokenizer):
    # os.environ['CUDA_VISIBLE_DEVICES'] = "5"
    # model_name = "daryl149/llama-2-7b-chat-hf"
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    # model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda")
    #
    # if from_peft_checkpoint:
    #     model = PeftModel.from_pretrained(model, from_peft_checkpoint, is_trainable=True)


    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
    outputs = model.generate(input_ids=input_ids,
                             max_new_tokens=max_tokens, return_dict_in_generate=True, output_scores=True,
                             output_hidden_states=True)

    tokenizer.batch_decode(outputs, skip_special_tokens=True)

    pdb.set_trace()

    return outputs

def Llama3ChatCompletion(model_name, prompt, max_tokens, model, tokenizer):
    # os.environ['CUDA_VISIBLE_DEVICES'] = "5"
    # model_name = "daryl149/llama-2-7b-chat-hf"
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    # model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda")
    # if from_peft_checkpoint:
    #     model = PeftModel.from_pretrained(model, from_peft_checkpoint, is_trainable=True)

    generation_kwargs = {
        "min_length": 1,
        "max_new_tokens": 100, 
        "top_k": 0.0,
        "top_p": 1.0,
        "temperature": 0.7,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
        "output_scores": True,
        "return_dict_in_generate": True,
        # "pad_token_id": tokenizer.pad_token_id
    }
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda") 
    outputs = model.generate(input_ids=input_ids,
                              **generation_kwargs)

    output = tokenizer.decode(outputs[0][0][input_ids.shape[-1]:], skip_special_tokens=True)
    print(output)

    return output

