# -*- coding: utf-8 -*-

import torch.nn as nn
from torch.nn import CrossEntropyLoss

# from transformers import RobertaModel, RobertaConfig
from transformers import LayoutLMForTokenClassification, AutoModelForTokenClassification
from transformers import AutoModelForQuestionAnswering, AutoConfig
from LMs.layoutlm_model import LayoutLMModel, LayoutLMForMaskedLM,LayoutLMOnlyMLMHead
from transformers.utils import ModelOutput
from transformers import AutoConfig



# this is CS masked language model, you cannot change the parameters from pre-trained anymore (unless you train from scratch)
class CSMaskedLM(nn.Module):
    def __init__(self, opt, freeze_bert=False):
        super(CSMaskedLM, self).__init__()
        self.opt = opt
        self.config = AutoConfig.from_pretrained(opt.layoutlm_large)

        self.cs_layoutlm = LayoutLMModel.from_pretrained(opt.csmodel)
        self.cls = LayoutLMOnlyMLMHead(self.config)

        # self.csmodel = LayoutLMForMaskedLM.from_pretrained(opt.layoutlm_large)
        print('Find the path of configuration: ', opt.layoutlm_large)
        # freeze the bert model
        if freeze_bert:
            for param in self.roberta.parameters():
                param.requires_grad = False
        

    def forward(self,input_ids, attention_mask, dist, direct, seg_width,seg_height,segmentation_ids,labels, **args):

        return_dict = self.config.use_return_dict

        outputs = self.cs_layoutlm(input_ids = input_ids, attention_mask = attention_mask, dist = dist, 
            direct = direct, seg_width=seg_width, seg_height=seg_height, segmentation_ids=segmentation_ids)
        # outputs = self.layoutlm(input_ids = input_ids, bbox = None, attention_mask = attention_mask, pixel_values = None, labels = labels)

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(
                prediction_scores.view(-1, self.config.vocab_size),
                labels.view(-1),
            )
        # else:
        #     print('==== why there is no labels?=======')

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return ModelOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            # last_hidden_states=sequence_output, # === added by me !!=====
            attentions=outputs.attentions,
        )

        return outputs

# class KeyValLinking(nn.Module):
#     def __init__(self, opt, freeze_bert=False):
#         super(KeyValLinking, self).__init__()
#         self.opt = opt
#         self.csmodel = LayoutLMModel.from_pretrained(opt.csmodel) 

#         self.predict = nn.Linear(self.opt.input_dim, 2),

#     def forward(cent_dict, edge_dict, link_label):
#         outputs1 = self.csmodel(cent_dict**)
#         outputs2 = self.csmodel(cent_dict**)

#         hidden_state1 = outputs1[0]
#         pooled_output1 = hidden_state1[:, 0] # further pool

#         hidden_state2 = outputs2[0]
#         pooled_output2 = hidden_state2[:, 0] # further pool
        
#         pair_pool = torch.cat([pooled_output1,pooled_output2],dim=-1)
#         return self.predict(pair_pool)


# tokenization 
class CSTokenClassifier(nn.Module):
    def __init__(self, opt, freeze_bert=False):
        super(CSTokenClassifier, self).__init__()
        self.opt = opt
        self.num_labels = opt.num_labels    # num_labels from dataset loading class
        self.csmodel = LayoutLMModel.from_pretrained(opt.csmodel)   # customized layoutlm/csmodel
        self.dropout = nn.Dropout(opt.dropout)
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
        # print(outputs.shape)
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

