# -*- coding: utf-8 -*-

import torch.nn as nn

# from transformers import RobertaModel, RobertaConfig
from transformers import LayoutLMForTokenClassification, AutoModelForTokenClassification
from transformers import AutoModelForQuestionAnswering, AutoConfig
from LMs.layoutlm_model import LayoutLMModel, LayoutLMForMaskedLM

# this is CS masked language model, you cannot change the parameters from pre-trained anymore (unless you train from scratch)
class CSModel(nn.Module):
    def __init__(self, opt, freeze_bert=False):
        super(CSModel, self).__init__()
        self.opt = opt
        # self.config = RobertaConfig.from_pretrained(opt.roberta_dir)
        # self.roberta = RobertaModel(self.config)
        # self.layoutlm = AutoModelForTokenClassification.from_pretrained(opt.layoutlm_dir, num_labels=opt.num_labels, label2id=opt.label2id, id2label=opt.id2label)
        self.csmodel = LayoutLMForMaskedLM.from_pretrained(opt.layoutlm_large)
        print('Find the path of configuration: ', opt.layoutlm_large)
        # freeze the bert model
        if freeze_bert:
            for param in self.roberta.parameters():
                param.requires_grad = False
        

    def forward(self,input_ids, attention_mask, dist, direct, seg_width,seg_height,segmentation_ids,labels, **args):
        outputs = self.csmodel(input_ids = input_ids, attention_mask = attention_mask, dist = dist, 
            direct = direct, seg_width=seg_width, seg_height=seg_height, segmentation_ids=segmentation_ids, labels = labels)
        # outputs = self.layoutlm(input_ids = input_ids, bbox = None, attention_mask = attention_mask, pixel_values = None, labels = labels)
        return outputs


# tokenization 
class CSTokenClassifier(nn.Module):
    def __init__(self, opt, freeze_bert=False):
        super(CSTokenClassifier, self).__init__()
        self.opt = opt
        self.num_labels = opt.num_labels    # num_labels from dataset loading class
        self.csmodel = LayoutLMModel.from_pretrained(opt.layoutlm_large)

        # hidden size = 1024
        self.classifier = nn.Linear(1024, self.num_labels)   

        print('Find the path of configuration: ', opt.layoutlm_large)
        # freeze the bert model
        if freeze_bert:
            for param in self.roberta.parameters():
                param.requires_grad = False

    def forward(self,input_ids, attention_mask, dist, direct, seg_width,seg_height,segmentation_ids,labels, **args):
        outputs = self.csmodel(input_ids = input_ids, attention_mask = attention_mask, dist = dist, 
            direct = direct, seg_width=seg_width, seg_height=seg_height, segmentation_ids=segmentation_ids)

        hidden_state = outputs[0]
        center_sequence_output = hidden_state[:, :192]    # take (batch_size, 192, dim)
        center_sequence_output = self.dropout(center_sequence_output)

        logits = self.classifier(center_sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return ModelOutput(
            loss=loss,
            logits = logits
        )

