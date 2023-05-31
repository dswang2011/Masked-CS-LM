from PIL import Image
import pytesseract
import pickle
import os
import json
from datasets import Dataset, load_from_disk
import numpy as np
import img_util
import cs_util
import tok_util
from datasets import Dataset,Features,Sequence, Value, Array2D, Array3D

def _load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    w, h = image.size
    return image, (w, h)


# double checked
def doc_to_segs(one_doc):
    texts,bboxes = [],[]
    word_nums = []

    seg_ids = one_doc['seg_ids']
    tokens = one_doc['tokens']
    boxes = one_doc['share_bboxes']   # shared boxes, share_bboxes, bboxes; here, it must be shared because it is seg oriented

    block_num = seg_ids[0]  # 11
    window_tokens = [tokens[0]]
    l = 0
    for i in range(1,len(seg_ids)):
        curr_id = seg_ids[i]
        if curr_id!=block_num:
            word_nums.append(len(window_tokens))
            text = ' '.join(window_tokens)
            texts.append(text)
            bboxes.append(boxes[l])
            # reset the params
            l = i
            block_num = curr_id
            window_tokens = [tokens[i]]
        else:
            window_tokens.append(tokens[i])
    word_nums.append(len(window_tokens))
    text = ' '.join(window_tokens)
    texts.append(text)
    bboxes.append(boxes[l])

    return texts,bboxes, word_nums


# one image to doc dict
def image_to_doc(image_path):
    '''
    rtype: return one_doc, where the bbox and h/w are normalized to 1000*1000
    '''
    # save to: (and will be extended with a 'shared_boxes')
    one_doc = {'tokens':[],'tboxes':[],'bboxes':[], 'block_ids':[],'image_path':image_path}

    image, size = _load_image(image_path)
    one_doc['size'] = size

    myconfig = r'--psm 11 --oem 3'
    data = pytesseract.image_to_data(image, config=myconfig, output_type='dict', timeout=10)

    texts = data['text']
    page_nums = data['page_num']
    block_nums = data['block_num']
    line_nums = data['line_num']
    x0s = data['left']
    y0s = data['top']
    hs = data['height'] # temporary use only, 
    ws = data['width']  # temporary use only

    # encoding = img_util.feature_extractor(image)
    # one_doc['image'] = encoding.pixel_values[0]    # image object, get the first one, cause there is only one!
    # one_doc['image'] = image
    for i,word in enumerate(texts):
        # token
        token = word.strip()
        if token=='': continue
        # height and width
        height, width = hs[i],ws[i]
        # coordinate
        x0 = x0s[i]
        y0 = y0s[i]
        x1 = x0 + width
        y1 = y0 + height
        # page, line, block, block_id
        page_num, line_num, block_num = page_nums[i],line_nums[i], block_nums[i]

        # produce one sample
        one_doc['tokens'].append(token)
        one_doc['tboxes'].append([x0,y0,x1,y1])
        one_doc['block_ids'].append(block_num)
    # add the shared box: bboxes
    one_doc = img_util._adjust_shared_bbox(one_doc)

    return one_doc


def get_img2doc_data(img_dir):
    res = {}    # a dict of dict, i.e., {docID_pageNO : {one_doc_info}}
    for doc_idx, file in enumerate(sorted(os.listdir(img_dir))):
        image_path = os.path.join(img_dir, file)
        one_doc = image_to_doc(image_path)
        docID_pageNO = file.replace(".png", "")
        res[docID_pageNO] = one_doc
        # print(one_doc)
        # if doc_idx>50:
        #     break
    return res


def get_question_pairs(base,split='val'):
    # from json of questions and answers
    file_path = os.path.join(base, split+'_v1.0.json')
    
    with open(file_path) as fr:
        data = json.load(fr)
    id2trip = {}
    for sample in data['data']:
        qID = sample['questionId']  # numeric e.g., 8366
        question = sample['question']
        # for test set, there is no answersr
        answers = []
        if 'answers' in sample.keys():
            answers = sample['answers']

        ucsf_doc_id = sample['ucsf_document_id']   # e.g.,: txpp0227
        ucsf_doc_page = sample['ucsf_document_page_no'] # e.g.,: 10
        docID_page = ucsf_doc_id + '_' + ucsf_doc_page
        trip_object = (docID_page, question, answers)
        id2trip[qID] = trip_object
    return id2trip


def wrap_and_save(base, split):
    # mydataset = Dataset.from_generator(generator_based_on_questions,gen_kwargs={'split':split, 'base':base})
    id2queryinfo, id2doc = produce_based_on_questions(base, split)
    print('q num:',len(id2queryinfo.keys()))
    print('doc num:', len(id2doc.keys()))
    output_to_pickle([id2queryinfo,id2doc],split+'.pickle')
    # save to disk


if __name__=='__main__':

    # step1: using OCR to extract seg texts and boxes from img
    file_path = '/home/ubuntu/air/vrdu/datasets/docvqa/test/documents/ffdw0217_13.png'
    # file_path = '/home/ubuntu/air/vrdu/datasets/images/imagesa/a/a/a/aaa06d00/50486482-6482.tif'
    one_doc = image_to_doc(file_path)
    texts, boxes, token_nums = doc_to_segs(one_doc)


    # step2: wrap to huggingface dataset
    final_dict = tok_util.doc_2_final_dict(boxes,texts,token_nums)
    print(final_dict.keys())
    
    # dataset = Dataset.from_dict(final_dict).with_format("torch")
    # print(dataset)
    # for i,row in enumerate(dataset):
    #     print('------')
    #     # print(texts[i])
    #     for k,v in row.items():
    #         print(k,v)
    #     if i>2:
    #         break


    # step3: models


    # edge_index, edge_attr = cs_util.rolling_neibor_matrix(boxes)
    # u,v = edge_index

    # for i,(u,v) in enumerate(zip(u,v)):
    #     print(u,'==v.s.==',v)
    #     print(texts[u], '==v.s.==', texts[v])
    #     print(boxes[u],'==v.s.==',boxes[v])
    #     print(edge_attr[i])
    #     print('----------')

    