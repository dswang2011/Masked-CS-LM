




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
        self.tokenizer()
        # get dataset from saved hf;

        # get proper masks and map to labels;


    def get_data():
        pass

    
    def map_one_match(encodings):
        inputs['labels'] = inputs.input_ids.detach.clone()


    # get labels as well;
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
        pad_id = tokenizer.pad_token_id # 0;    roberta = 1
        mask_id = tokenizer.mask_token_id   # ? roberta = 50264

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
