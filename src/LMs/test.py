import torch
from layoutlm_model import LayoutLMModel, LayoutLMForMaskedLM

path = '/Users/yulong/Documents/Projects/Huggingface/distilbert-base-uncased'
model = LayoutLMForMaskedLM.from_pretrained(path)