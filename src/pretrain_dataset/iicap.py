

from datasets import load_dataset ,Features, Sequence, Value, Array2D, Array3D
from datasets.features import ClassLabel
from transformers import AutoProcessor
import json
import os

class IICAP:
    def __init__(self,opt) -> None:    
        self.opt = opt

        self.tokenizer = AutoTokenizer.from_pretrained(opt.layoutlm_dir)    #tokenizer
        assert isinstance(self.tokenizer, transformers.PreTrainedTokenizerFast) # get sub



    def get_dataset():
        pass


    def get_masks():
        pass