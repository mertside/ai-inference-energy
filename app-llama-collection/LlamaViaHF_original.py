#! /usr/bin/env python

# make sure you're logged in with `huggingface-cli login`
import torch
from transformers import pipeline

pipeline = pipeline(task="text-generation", model="huggyllama/llama-7b", torch_dtype=torch.float16, device=0)
pipeline("Plants create energy through a process known as")
