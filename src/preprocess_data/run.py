
import datasets
import tesseract4img
import tok_util
from datasets import Dataset,Features,Sequence, Value, Array2D, Array3D
import os
import random


# step1: get all images from a folder
images = '/home/ubuntu/air/vrdu/datasets/images'

def get_file_list():    # .tif files
    res = []
    train = '/home/ubuntu/air/vrdu/datasets/labels/train.txt'
    # val = '/home/ubuntu/air/vrdu/datasets/labels/val.txt'
    # train = '/home/ubuntu/air/vrdu/datasets/labels/test.txt'
    with open(train,'r',encoding='utf8') as fr:
        data = fr.readlines()
    for row in data:
        path = row.split(' ')[0]
        image_path = os.path.join(images,path)
        res.append(image_path)
    return res


# step2: shuffle and split into 20 subsets
def split(input_list, n):
    k, m = divmod(len(input_list), n)
    return (input_list[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))


# step3: for each subset images:
# 
def generate_and_save(set_id, imgs):
    saveto = '/home/ubuntu/air/vrdu/datasets/rvl_pretrain_datasets/' + str(set_id) +'_bert.hf'
    all_doc_dataset = []

    for i,img_path in enumerate(imgs):
        print(str(i), 'currently process:',img_path)
        one_doc = tesseract4img.image_to_doc(img_path)
        # did not get anything
        if not one_doc or not one_doc['tokens']:
            print('detect empty content')
            continue
        texts, boxes, token_nums = tesseract4img.doc_to_segs(one_doc)
        final_dict = tok_util.doc_2_final_dict(boxes,texts,token_nums)
        doc_dataset = Dataset.from_dict(final_dict) # with_format("torch")

        all_doc_dataset.append(doc_dataset)
        
    final_dataset = datasets.concatenate_datasets(all_doc_dataset)
    final_dataset.save_to_disk(saveto)
    print(final_dataset)

    return saveto

if __name__=="__main__":
    files = get_file_list()


    random.Random(3).shuffle(files)

    subsets = list(split(files, 20))
    for i,subset in enumerate(subsets):
        print(len(subset))
        saveto = generate_and_save(i,subset)
        # print('saved to:', saveto)
