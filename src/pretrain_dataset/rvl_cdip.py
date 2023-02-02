from datasets import load_from_disk, Features, Sequence, Value

from transformers import RobertaTokenizer
import torch


# class MyDataset(torch.utils.data.Dataset):
#     def __init__(self, encodings):
#         self.encodings = encodings
#     def __getitem__(self, idx):
#         return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
#     def __len__(self):
#         return len(self.encodings.input_ids)


class RVLCDIP:
    def __init__(self,opt,start_chunk=1) -> None:    
        self.opt = opt
        
        # tokenizer
        self.tokenizer = RobertaTokenizer.from_pretrained(opt.roberta_dir)

        # get dataset from saved hf;
        self.train_dataset = self.get_data(opt.rvl_cdip).with_format("torch")
        print(self.train_dataset)
        print(self.train_dataset[0])
        # get proper masks and map to labels;
        self.masked_train_dataset = self.masked_inputs(self.train_dataset).with_format("torch")
        print(self.masked_train_dataset[0])

    def get_data(self, hf_path):
        return load_from_disk(hf_path)

    
    # def map_one_match(encodings):
    #     inputs['labels'] = inputs.input_ids.detach.clone()


    # get labels as well;
    def masked_inputs(self,dataset, mask_ratio=0.3):
        '''
        return: selections, batch_inputs
        '''
        cls_id = self.tokenizer.cls_token_id # 101; roberta = 0  or <s>
        sep_id = self.tokenizer.sep_token_id # 102; roberta = 2  or </s>
        pad_id = self.tokenizer.pad_token_id # 0;    roberta = 1
        mask_id = self.tokenizer.mask_token_id   # ? roberta = 50264

        # mapping
        def map_label(batch):
            batch['labels'] = batch['input_ids'].clone()

            att_mask = batch['attention_mask']

            # step1: random a (batch_size, sequence_num) 
            rand_mat = torch.rand(len(batch['input_ids']),512)
            # create mask array
            threshold = (rand_mat < mask_ratio)
            mask_arr = threshold * (batch['input_ids'] != cls_id) * \
                (batch['input_ids'] != sep_id) * (batch['input_ids'] != pad_id) * (att_mask!=0)
            # now we take the indices of each True value, for each vector.
            selects = []  # positions that are masked
            for i in range(len(batch['input_ids'])):
                select = torch.flatten(mask_arr[i].nonzero()).tolist()
                select = [item for item in select if item < 192]
                if not select: 
                    select = [1]
                selects.append(select)
            # Step2: Apply these indices to each respective row in input_ids, assigning [MASK] positions as 103.
            for i in range(len(batch['input_ids'])):
                # print('ex:',example['input_ids'][i])
                # print('selected positions:',selects[i])
                # print('maskid:',mask_id)
                batch['input_ids'][i,selects[i]] = mask_id
            return batch 
        
        # step0: copy original input_ids as the labels
        features = Features({
            'input_ids': Sequence(feature=Value(dtype='int64')),
            'attention_mask': Sequence(Value(dtype='int64')),
            'dist': Sequence(Value(dtype='int64')),
            'direct': Sequence(Value(dtype='int64')),
            'seg_width': Sequence(Value(dtype='int64')),
            'seg_height': Sequence(Value(dtype='int64')),
            'labels': Sequence(feature=Value(dtype='int64'))
        })
        dataset = dataset.map(map_label, batched=True, features = features)

        return dataset
