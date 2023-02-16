
from typing import Tuple
import math
import numpy as np
from math import sqrt

# import tesseract4img
    

def _rule_polar(rect_src : list, rect_dst : list) -> Tuple[int, int]:
    """Compute distance and direction from src to dst bounding boxes
    Args:
        rect_src (list) : source rectangle coordinates
        rect_dst (list) : destination rectangle coordinates
    
    Returns:
        tuple (ints): distance and direction
    """
    
    # check relative position
    left = (rect_dst[2] - rect_src[0]) < 0 # left-top point
    bottom = (rect_src[3] - rect_dst[1]) < 0   # 
    right = (rect_src[2] - rect_dst[0]) < 0
    top = (rect_dst[3] - rect_src[1]) < 0
    
    vp_intersect = (rect_src[0] <= rect_dst[2] and rect_dst[0] <= rect_src[2]) # True if two rects "see" each other vertically, above or under
    hp_intersect = (rect_src[1] <= rect_dst[3] and rect_dst[1] <= rect_src[3]) # True if two rects "see" each other horizontally, right or left
    rect_intersect = vp_intersect and hp_intersect 

    if rect_intersect:
        return 0, 0
    elif top and left:
        a, b = (rect_dst[2] - rect_src[0]), (rect_dst[3] - rect_src[1])
        return int(sqrt(a**2 + b**2)), 4
    elif left and bottom:
        a, b = (rect_dst[2] - rect_src[0]), (rect_dst[1] - rect_src[3])
        return int(sqrt(a**2 + b**2)), 6
    elif bottom and right:
        a, b = (rect_dst[0] - rect_src[2]), (rect_dst[1] - rect_src[3])
        return int(sqrt(a**2 + b**2)), 8
    elif right and top:
        a, b = (rect_dst[0] - rect_src[2]), (rect_dst[3] - rect_src[1])
        return int(sqrt(a**2 + b**2)), 2
    elif left:
        return int(rect_src[0] - rect_dst[2]), 5
    elif right:
        return int(rect_dst[0] - rect_src[2]), 1
    elif bottom:
        return int(rect_dst[1] - rect_src[3]), 7
    elif top:
        return int(rect_src[1] - rect_dst[3]), 3


def _fully_connected_matrix(bboxs):
    pair_lookup = {}
    for i in range(len(bboxs)):
        for j in range(len(bboxs)):
            box1, box2 = bboxs[i],bboxs[j]
            dist,direct = _rule_polar(box1, box2)
            # direct = angle //45 
            pair_lookup[(i,j)] = (dist,direct)
    return pair_lookup
def _eight_neibs(idx, N,pair_lookup):
    direct2near = {}    # idx : (dist, direct, index)
    for neib_idx in range(N):
        if idx == neib_idx: continue  # skip itself
        dist,direct = pair_lookup[(idx,neib_idx)]
        if direct in direct2near.keys():
            if dist<direct2near[direct][0]:
                direct2near[direct] = (dist, neib_idx)
        else:
            direct2near[direct] = (dist, neib_idx)
    return direct2near

def rolling_neibor_matrix(bboxs):
    pair_lookup = _fully_connected_matrix(bboxs)
    u,v = [],[]
    # dists, directs = [],[]
    edge_index = []
    edge_attr = []
    for idx in range(len(bboxs)):
        direct2near = _eight_neibs(idx, len(bboxs), pair_lookup)
        for direct, (dist,direct,neib_idx) in direct2near.items():
            u.append(idx)
            v.append(neib_idx)
            # dist.append(dist)
            # directs.append(direct)
            edge_attr.append([dist,direct])
    edge_index = [u,v]
    return edge_index, edge_attr


def rolling_8neibors(bboxs):
    pair_lookup = _fully_connected_matrix(bboxs)
    neibors = []
    for idx in range(len(bboxs)):
        direct2near = _eight_neibs(idx, len(bboxs), pair_lookup)
        # value = (dist, direct, neib_idx)
        neibors.append(direct2near)
    return neibors

# from PIL import Image, ImageDraw, ImageFont
# import IPython.display as display

if __name__=='__main__':
    file_path = '/home/ubuntu/air/vrdu/datasets/docvqa/test/documents/ffdw0217_13.png'
    # file_path = '/home/ubuntu/air/vrdu/datasets/images/imagesa/a/a/a/aaa06d00/50486482-6482.tif'
    image = Image.open(file_path)
    image = image.convert("RGB")
    draw = ImageDraw.Draw(image, "RGBA")


    one_doc = tesseract4img.image_to_doc(file_path, box_norm=False)
    texts, boxes, token_nums = tesseract4img.doc_to_segs(one_doc)

    neibs = rolling_8neibors(boxes)

    # take one
    idx = 5
    direct2near = neibs[idx]
    c_box = boxes[idx]
    print(texts[idx], c_box)
    # draw center
    draw.rectangle(c_box, outline='red', width=3)
    for direct, (dist, neib_idx) in direct2near.items():
        n_box = boxes[neib_idx]
        print(direct, ':',texts[neib_idx], n_box)
        draw.rectangle(n_box, outline='orange', width=2)
    
    # image.show()
    image = image.save("temp.jpg")



