import torch
from layoutlm_model import LayoutLMModel

path = '/Users/yulong/Documents/Projects/Huggingface/distilbert-base-uncased'
model = LayoutLMModel.from_pretrained(path)