# -*- coding: utf-8 -*-

import torch.nn as nn

# from transformers import RobertaModel, RobertaConfig
from transformers import LayoutLMForTokenClassification, AutoModelForTokenClassification
from transformers import AutoModelForQuestionAnswering
from LMs.layoutlm_model import LayoutLMModel, LayoutLMForMaskedLM


class CSModel(nn.Module):
    def __init__(self, opt, freeze_bert=False):
        super(CSModel, self).__init__()
        self.opt = opt
        # self.config = RobertaConfig.from_pretrained(opt.roberta_dir)
        # self.roberta = RobertaModel(self.config)
        # self.layoutlm = AutoModelForTokenClassification.from_pretrained(opt.layoutlm_dir, num_labels=opt.num_labels, label2id=opt.label2id, id2label=opt.id2label)
        self.csmodel = LayoutLMForMaskedLM.from_pretrained(opt.roberta_dir)
        # freeze the bert model
        if freeze_bert:
            for param in self.roberta.parameters():
                param.requires_grad = False
        

    def forward(self,input_ids, attention_mask, dist, direct, seg_width,seg_height,segmentation_ids,labels, **args):
        outputs = self.csmodel(input_ids = input_ids, attention_mask = attention_mask, dist = dist, 
            direct = direct, seg_width=seg_width, seg_height=seg_height, segmentation_ids=segmentation_ids, labels = labels)
        # outputs = self.layoutlm(input_ids = input_ids, bbox = None, attention_mask = attention_mask, pixel_values = None, labels = labels)
        return outputs

