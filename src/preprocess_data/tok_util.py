import torch
from transformers import RobertaTokenizer,RobertaForMaskedLM
import cs_util

from transformers import LayoutLMTokenizer
from transformers import AutoTokenizer

# tokenizer = RobertaTokenizer.from_pretrained('/home/ubuntu/resources/roberta.base.squad')
# tokenizer = LayoutLMTokenizer.from_pretrained('/home/ubuntu/air/vrdu/models/layoutlmv1.large')
tokenizer = LayoutLMTokenizer.from_pretrained("/home/ubuntu/air/vrdu/models/layoutlmv1.large")
# tf_tokenizer = TFBertTokenizer.from_tokenizer(tokenizer)


def _empty_s_encod():
    '''
    rtype: input_id_tensor (len=40), attention_mask_tensor (len=40)
    '''
    neib_pad = tokenizer('', return_tensors='pt', max_length=40, truncation=True, padding='max_length')
    neib_pad.input_ids[:,0] = tokenizer.sep_token_id
    return neib_pad.input_ids[0],neib_pad.attention_mask[0]


# one_doc batch: used to turn a doc segments, into pair dict;
def pair_encoding(texts, tokenizer):
    c_encodings = tokenizer(texts, return_tensors='pt', max_length=192, truncation=True, padding='max_length')
    s_encodings = tokenizer(texts, return_tensors='pt', max_length=40, truncation=True, padding='max_length')
    s_encodings.input_ids[:, 0] = tokenizer.sep_token_id
    return c_encodings,s_encodings

def cs_encoding(c_encodings, s_encodings, neibors):
    # empty pad
    emp_inp, emp_att = _empty_s_encod()

    res = {'input_ids':[],'attention_mask':[],'dist':[],'direct':[]}
    for idx, neib in enumerate(neibors):    # (dist, direct, neib_idx)
        cs_inps, cs_atts = [c_encodings.input_ids[idx]],[c_encodings.attention_mask[idx]]
        dists, directs = [0.0],[0]
        for direct in range(1,9):
            if direct not in neib:
                cs_inps.append(emp_inp)
                cs_atts.append(emp_att)

                dists.append(1000)
                directs.append(direct)
            else:
                dist, direct, neib_idx = neib[direct]
                cs_inps.append(s_encodings.input_ids[neib_idx])
                cs_atts.append(s_encodings.attention_mask[neib_idx])

                dists.append(dist)
                directs.append(direct)

        comb_inp = torch.cat(cs_inps, dim=0)
        comb_att = torch.cat(cs_atts, dim=0)

        res['input_ids'].append(comb_inp)
        res['attention_mask'].append(comb_att)
        res['dist'].append(dists)
        res['direct'].append(directs)
    return res


def get_relative_info(boxes,neibors, token_nums):
    seg_widths, seg_heights = [],[]
    for idx, neib in enumerate(neibors):    # (dist, direct, neib_idx)
        w,h = boxes[idx][2]-boxes[idx][0], boxes[idx][3]-boxes[idx][1]
        s_w, s_h = [w],[h]

        for direct in range(1,9):
            if direct not in neib:   
                s_w.append(1000)
                s_h.append(1000)
            else:
                dist, direct, neib_idx = neib[direct]
                nw,nh = boxes[neib_idx][2]-boxes[neib_idx][0], boxes[neib_idx][3]-boxes[neib_idx][1]
                s_w.append(nw)
                s_h.append(nh)
        seg_widths.append(s_w)
        seg_heights.append(s_h)
    return seg_widths, seg_heights

def doc_2_final_dict(boxes,texts,token_nums):
    neibors = cs_util.rolling_8neibors(boxes)
    c_encodings, s_encodings = pair_encoding(texts,tokenizer)
    # get info 1
    final_doc_dict = cs_encoding(c_encodings, s_encodings, neibors)
    # get info 2
    ws,hs = get_relative_info(boxes, neibors, token_nums)
    final_doc_dict['seg_width'] = ws
    final_doc_dict['seg_height'] = hs

    return final_doc_dict


