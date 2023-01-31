import os
import argparse
from torch.utils.data import DataLoader

import torch
import pickle
from utils.params import Params
# from torch_geometric.transforms import NormalizeFeatures
import pretrain_dataset
from LMs import trainer
# from utils import util_trainer 


from transformers import RobertaTokenizer,RobertaForMaskedLM
from prepare_dataset import cs_util


def _empty_s_encod():
    '''
    rtype: input_id_tensor (len=40), attention_mask_tensor (len=40)
    '''
    neib_pad = tokenizer('', return_tensors='pt', max_length=40, truncation=True, padding='max_length')
    neib_pad.input_ids[:,0] = tokenizer.sep_token_id
    print(neib_pad.input_ids[0])
    print(neib_pad.attention_mask[0])
    return neib_pad.input_ids[0],neib_pad.attention_mask[0]


# one_doc batch: used to turn a doc segments, into pair dict;
def pair_encoding(texts, tokenizer):
    c_encodings = tokenizer(texts, return_tensors='pt', max_length=192, truncation=True, padding='max_length')
    s_encodings = tokenizer(texts, return_tensors='pt', max_length=40, truncation=True, padding='max_length')
    s_encodings.input_ids[:, 0] = tokenizer.sep_token_id
    return c_encodings,s_encodings


def doc_2_encoding(boxes,texts):
    neibors = cs_util.rolling_8neibors(boxes)
    c_encodings, s_encodings = pair_encoding(texts,tokenizer)
    final_doc_encoding = cs_encoding(c_encodings, s_encodings, neibors)
    return final_doc_encoding


def cs_encoding(c_encodings, s_encodings, neibors):
    # empty pad
    emp_inp, emp_att = _empty_s_encod()

    for idx, neib in enumerate(neibors):    # (dist, direct, neib_idx)
        c_encod = c_encodings[idx]
        s_inps, s_atts = [],[]
        for direct in range(1,9):
            if direct not in neib:
                s_inps.append(emp_inp)
                s_atts.append(emp_att)
            else:
                dist, direct, neib_idx = neib[direct]
                s_inps.append(s_encodings.input_ids[neib_idx])
                s_atts.append(s_encodings.attention_mask[neib_idx])
        c_encod.input_ids = torch.cat([c_encod.input_ids] + s_inps ,-1)
        c_encod.attention_mask = torch.cat([c_encod.attention_mask] + s_atts ,-1)
    return c_encodings


# s 50 sequences;
def cs_encoding(c_encod, s_encods):
    cs_encodings = {}
    for k,v in c_encod.items():
        cs_encodings[k] = torch.cat((v, s_encods[k]),-1)
    return cs_encodings


def masked_inputs(batch_encodings,tokenizer, mask_ratio=0.3):
    '''
    return: selections, batch_inputs
    '''
    # step1: random a (batch_size, sequence_num) 
    rand_mat = torch.rand(batch_encodings.input_ids.shape)
    # create mask array
    threshold = (rand_mat < mask_ratio)

    cls_id = tokenizer.cls_token_id # 101; roberta = 0  or <s>
    sep_id = tokenizer.sep_token_id # 102; roberta = 2  or </s>
    
    bos_id = tokenizer.bos_token_id # roberta = 0
    eos_id = tokenizer.eos_token_id # roberta = 2

    pad_id = tokenizer.pad_token_id # 0;    roberta = 1
    mask_id = tokenizer.mask_token_id   # ? roberta = 50264


    print('4 nums:',cls_id, sep_id, pad_id, mask_id, bos_id, eos_id)

    mask_arr = threshold * (batch_encodings.input_ids != cls_id) * \
            (batch_encodings.input_ids != sep_id) * (batch_encodings.input_ids != pad_id)

    # now we take the indices of each True value, for each vector.
    selects = []  # positions that are masked
    for i in range(batch_encodings.input_ids.shape[0]):
        select = torch.flatten(mask_arr[i].nonzero()).tolist()
        select = [item for item in select if item < 192]
        if not select: 
            select = [1]
        selects.append(select)

    # Step2: Apply these indices to each respective row in input_ids, assigning [MASK] positions as 103.
    for i in range(batch_encodings.input_ids.shape[0]):
        batch_encodings.input_ids[i, selects[i]] = mask_id 
        reconstructed_answer = tokenizer.decode(batch_encodings.input_ids[i])
        print("Reconstructed answer:", reconstructed_answer)

    return selects, batch_encodings


def parse_args(config_path):
    parser = argparse.ArgumentParser(description='run the model')
    parser.add_argument('--config', dest='config_file', default = config_path)
    return parser.parse_args()

if __name__=='__main__':

    # Section 1, parse parameters
    args = parse_args('config/lm.ini') # from config file
    params = Params()   # put to param object
    params.parse_config(args.config_file)
    params.config_file = args.config_file

    params.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print('Using device:', params.device)

    # section 2, load data; prepare output_dim/num_labels, id2label, label2id for section3; 
    # params.train_split, params.test_split = 'val','test'
    # params.start_chunk = 1
    # mydata = pretrain_dataset.setup(params)
    
    tokenizer = RobertaTokenizer.from_pretrained(params.roberta_dir)
    model = RobertaForMaskedLM.from_pretrained(params.roberta_dir)
    texts = ["test some text for language model.", "another test example for it","i try some manual <mask> and some <mask> so on"]

    # section 3, objective function and output dim/ move to trainer
    # params.criterion = util_trainer.get_criterion(params)

    # section 4, model, loss function, and optimizer
    # if bool(params.continue_train):
    #     print('continue based on:', params.continue_with_model)
    #     model_params = pickle.load(open(os.path.join(params.continue_with_model,'config.pkl'),'rb'))
    #     model = LMs.setup(model_params).to(params.device)
    #     model.load_state_dict(torch.load(os.path.join(params.continue_with_model,'model')))
    # else:
    #     model = LMs.setup(params).to(params.device)


    c_encodings, s_encodings = pair_encoding(texts,tokenizer)

    doc_2_encoding(boxes,texts)

    # print(cs_encoding)

    # encodings = tokenizer(texts)
    # print(s_encodings)

    # # encodings['labels'] = encodings.input_ids.clone().detach()

    # selectons,encodings = masked_inputs(s_encodings,tokenizer=tokenizer)

    # print('orig inputs', s_encodings.input_ids)
    # # print('labels', encodings['labels'])
    # print('selections', selectons)
    # print('new_inputs', encodings.input_ids)

    # section 5, train and evaluate, train each 10000 for each part;
    # 5.1 save to folder
    # params.dir_path = trainer.create_save_dir(params)    # prepare dir for saving best models

    # best_f1 = trainer.train(params, model, mydata)
    # inferencer.inference(params,model,mydata,'v3_base_benchmark_Jan26_'+str(params.start_chunk) +'.json')

    # for chunk in range(2,params.train_part+1):
    #     mydata.adjust(chunk)
    #     best_f1,best_loss = trainer.train(params, model, mydata)
    #     inferencer.inference(params,model,mydata,'v3_base_benchmark_Jan26_'+str(chunk)+'.json')

    #     print('best f1:', best_f1)
    #     print('best loss:', best_loss)

    # section 6, inference only (on test_dataset)
    # inferencer.inference(params,model,mydata,'v3_base_benchmark_Jan_1pm.json')


