
import datasets
import tesseract4img
import tok_util
from datasets import Dataset,Features,Sequence, Value, Array2D, Array3D
import os
import random


tgt_prob = {
    '0':0.15,
    '1':1.0,
    '2':0.15,
    '3':0.15,
    '4':0.25,
    '5':0.45,
    '6':0.2,
    '7':0.55,
    '8':0.1,
    '9':0.15,
    '10':0.7,
    '11':0.95,
    '12':0.15,
    '13':0.7,
    '14':0.3,
    '15':0.15,
}


# step1: get all images from a folder
images = '/home/ubuntu/air/vrdu/datasets/images'

def get_file_list():    # .tif files
    res = []
    labels = []
    train = '/home/ubuntu/air/vrdu/datasets/labels/train.txt'
    # val = '/home/ubuntu/air/vrdu/datasets/labels/val.txt'
    # train = '/home/ubuntu/air/vrdu/datasets/labels/test.txt'
    with open(train,'r',encoding='utf8') as fr:
        data = fr.readlines()
    for row in data:
        strs = row.split(' ')
        path = strs[0]
        label = strs[1].strip()
        image_path = os.path.join(images,path)
        res.append(image_path)
        labels.append(label)
    return res, labels


# step2: shuffle and split into 20 subsets
def split(input_list, n):
    k, m = divmod(len(input_list), n)
    return (input_list[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))


# step3: for each subset images:
# 
def generate_and_save(set_id, imgs):
    saveto = '/home/ubuntu/air/vrdu/datasets/rvl_pretrain_datasets/weighted_' + str(set_id) +'_cs.hf'
    all_doc_dataset = []

    for i,img_path in enumerate(imgs):
        print(str(i), 'currently process:',img_path)
        one_doc = tesseract4img.image_to_doc(img_path)
        # did not get anything
        if not one_doc or not one_doc['tokens']:
            print('detect empty content')
            continue
        texts, boxes, token_nums = tesseract4img.doc_to_segs(one_doc)
        final_dict,_ = tok_util.doc_2_final_dict(boxes,texts)   # return: dict, neib
        doc_dataset = Dataset.from_dict(final_dict) # with_format("torch")

        all_doc_dataset.append(doc_dataset)
        
    final_dataset = datasets.concatenate_datasets(all_doc_dataset)
    final_dataset.save_to_disk(saveto)
    print(final_dataset)

    return saveto

def get_ratioly_sampled(imgs,labels):
    sub_imgs, sub_labels = [],[]
    for img,label in zip(imgs,labels):
        prob = random.random()
        if prob > tgt_prob[label]: continue
        sub_imgs.append(img)
        sub_labels.append(label)
    return sub_imgs, sub_labels

if __name__=="__main__":
    files, labels = get_file_list()
    sub_imgs, sub_labels = get_ratioly_sampled(files,labels)    # sampled
    file_part5, label_part5 = split(sub_imgs,5), split(sub_labels,5)

    for i, (sub_files, sub_labels) in enumerate(zip(file_part5, label_part5)):
        print(len(sub_files))
        saveto = generate_and_save(i,sub_files)
        # print('saved to:', saveto)
