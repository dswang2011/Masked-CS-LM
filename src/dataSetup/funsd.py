import sys
sys.path.append('../')
import os
from datasets import load_dataset, Features, Sequence, Value
from datasets import Dataset, DatasetDict
from utils.params import Params
import json
from preprocess_data import img_util, tok_util, cs_util
from transformers import AutoTokenizer
import datasets

class FUNSD:
    def __init__(self,opt):    
        self.opt = opt
        
        self.label_col_name = "labels"
        self.pad_token_label = -100
        self.id2label = {0: 'O', 1: 'B-HEADER', 2: 'I-HEADER', 3: 'B-QUESTION', 4: 'I-QUESTION', 5: 'B-ANSWER', 6: 'I-ANSWER'}
        self.label2id = {v:k for k,v in self.id2label.items()}
        # set glob param
        self.opt.num_labels = len(self.id2label.keys()) # for token classification usage;
        self.opt.label_list = list(self.id2label.values())  #

        self.tokenizer = AutoTokenizer.from_pretrained(opt.layoutlm_large) #layoutLMv1.large tokenizer

        # 1 load raw dataset; 2 map to trainable dataset
        # 1 get raw data
        self.train_test_dataset = self.get_train_test()


        # use raw data to encode & normalize labels
        # opt.id2label, opt.label2id, opt.label_list = self._get_label_map(self.dataset)
        # opt.num_labels = len(opt.label_list)
        
        # encode label class (target)
        # self.dataset['train'] = self.encode_class(self.dataset['train'])
        # self.dataset['test'] = self.encode_class(self.dataset['test'])

        # 2 get
        # self.cs_maps(self.train_test_dataset['train'])
        self.trainable_dataset = self.get_trainable(self.train_test_dataset)
        # 3 encode label column
        # dataset = dataset.cast_column(self.label_col_name, dst_feat)

        

    def encode_class(self,dataset):
        dst_feat = ClassLabel(names = self.opt.label_list)
        # Mapping Labels to IDs
        def map_label2id(example):
            example[self.label_col_name] = [dst_feat.str2int(ner_label) for ner_label in example[self.label_col_name]]
            return example
        dataset = dataset.map(map_label2id, batched=True)
        # type to ClassLabel object
        # dataset = dataset.cast_column(self.label_col_name, dst_feat)
        return dataset

    # 1 load raw dataset; one doc per img (not multiple pages)
    def get_train_test(self):
        train = Dataset.from_dict(self._load_samples(self.opt.funsd_train))
        test = Dataset.from_dict(self._load_samples(self.opt.funsd_test))
        return DatasetDict({
            'train':train,
            'test':test
        })
    def _load_samples(self,base_dir):
        ann_dir = os.path.join(base_dir, "adjusted_annotations")
        img_dir = os.path.join(base_dir, "images")
        
        batch_seg_texts = []
        batch_seg_labels = []
        batch_seg_boxs = []
        docIDs = []
        for doc_idx, file in enumerate(sorted(os.listdir(ann_dir))):
            # print('---doc id:---',doc_idx)
            docIDs.append(doc_idx)
            seg_texts = []
            labels = []
            shared_boxes = []

            file_path = os.path.join(ann_dir, file)
            with open(file_path, "r", encoding="utf8") as f:
                data = json.load(f)
            image_path = os.path.join(img_dir, file)
            image_path = image_path.replace("json", "png")
            image, size = img_util._load_image(image_path)
            seg_id = 0
            for item in data["form"]:
                cur_line_bboxes = []
                words, label = item["words"], item["label"]
                text = item['text']
                if text.strip()=='': continue
                words = [w for w in words if w["text"].strip() != ""]
                if len(words) == 0:
                    continue
                
                seg_texts.append(text)
                labels.append(label)

                # shared box
                for w in words:
                    cur_line_bboxes.append(img_util._normalize_bbox(w["box"], size))
                cur_line_bboxes = img_util._get_line_bbox(cur_line_bboxes)
                shared_boxes.append(cur_line_bboxes[0])
            # batch
            batch_seg_boxs.append(shared_boxes)
            batch_seg_texts.append(seg_texts)
            batch_seg_labels.append(labels)


        return {"id": docIDs, "seg_texts": batch_seg_texts, "shared_boxes": batch_seg_boxs, "labels": batch_seg_labels}


    def get_trainable(self, train_test):

        features = Features({
            'input_ids': Sequence(feature=Value(dtype='int64')),
            'attention_mask': Sequence(Value(dtype='int64')),
            'dist': Sequence(Value(dtype='int64')),
            'direct': Sequence(Value(dtype='int64')),
            'seg_width': Sequence(Value(dtype='int64')),
            'seg_height': Sequence(Value(dtype='int64')),
            'labels': Sequence(feature=Value(dtype='int64'))
        })
        train = self._cs_producer(train_test['train']).map(batched=True, features=features)
        test = self._cs_producer(train_test['test']).map(batched=True, features=features)

        return DatasetDict({
            "train" : train.with_format("torch") , 
            "test" : test.with_format("torch") 
        })

    # produce by maping data
    def _cs_producer(self,batch):
        all_batch = []  # 
        for doc in batch:
            seg_texts = doc['seg_texts']
            seg_labels = doc['labels']
            shared_boxes = doc['shared_boxes']

            final_doc_dict = tok_util.doc_2_final_dict(shared_boxes,seg_texts,0)
            extended_labels = [self._extend_label(seg_text,seg_label) for seg_text,seg_label in zip(seg_texts,seg_labels)]
            final_doc_dict['labels'] = extended_labels
            # ['input_ids', 'attention_mask', 'dist', 'direct', 'seg_width', 'seg_height', 'labels']
            # print(final_doc_dict.keys())
            # for k,v in final_doc_dict.items():
            #     print(k, ':',len(v[0]))
            doc_dataset = Dataset.from_dict(final_doc_dict) # produce one dataset for each doc
            all_batch.append(doc_dataset)
        batch_dataset = datasets.concatenate_datasets(all_batch)    # concatenate all doc datasets
        return batch_dataset

    
    # get a seqeunce of labels for a single segment text;
    def _extend_label(self,seg_text,label):
        seg_words = seg_text.split(' ')
        token_labels = []                  
        # extend for each word 
        if label == 'other':
            id_other = self.label2id['O']
            for w in seg_words:
                word_tokens = self.tokenizer.tokenize(w)
                token_labels.extend([id_other] + [self.pad_token_label] * (len(word_tokens) - 1))
        else:
            id_begin = self.label2id["B-"+label.upper()]
            id_inside = self.label2id["I-"+label.upper()]
            # first word
            word_tokens = self.tokenizer.tokenize(seg_words[0])
            token_labels.extend([id_begin] + [self.pad_token_label] * (len(word_tokens) - 1))
            # remaining words
            for w in seg_words[1:]:
                word_tokens = self.tokenizer.tokenize(w)
                token_labels.extend([id_inside] + [self.pad_token_label] * (len(word_tokens) - 1))
        padded_label = [-100] + token_labels + [-100]*(191-len(token_labels))
        return padded_label


    # def _my_tokenizer(batch_words,batch_boxes):
    #     texts = []
    #     bboxes = []
    #     labels = []
    #     for words, boxes in zip(batch_words,batch_boxes):
    #         token_ids = []
    #         token_boxes = []
    #         token_labels = []
    #         for word, box in zip(words, normalized_word_boxes):
    #             word_tokens = self.tokenizer.tokenize(word)
    #             token_ids.extend(word_tokens)
    #             token_boxes.extend([box] * len(word_tokens))
    #             token_labels.extend([label] + [self.pad_token_label] * (len(word_tokens) - 1))
            
    #     encoding = self.tokenizer(texts, return_tensors="pt")
    #     encoding['bbox'] = torch.tensor(token_boxes)
    #     return encoding

if __name__=='__main__':
    params = Params() 
    params.funsd_train = '/home/ubuntu/air/vrdu/datasets/FUNSD/training_data/'
    params.funsd_test = '/home/ubuntu/air/vrdu/datasets/FUNSD/testing_data/'
    params.layoutlm_large = '/home/ubuntu/air/vrdu/models/layoutlmv1.base'
    funsd = FUNSD(params)
    train_dataset = funsd.train_test_dataset['train']
    '''
    Dataset({
        features: ['id', 'tokens', 'bboxes', 'ner_tags', 'image', 'seg_ids'],
        num_rows: 149
    })
    '''
    print(train_dataset)


