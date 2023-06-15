import copy
import json
from collections import defaultdict
import random
from random import uniform    
import argparse
import os
coco_class = (1,2,3,4,5,6,7,8,9,10,11,13,14,15,16,17,18,19,20,\
              21,22,23,24,25,27,28,31,32,33,34,35,36,37,38,39,40,\
              41,42,43,44,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,\
              61,62,63,64,65,67,70,72,73,74,75,76,77,78,79,80,\
              81,82,84,85,86,87,88,89,90)
voc_class = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19)

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--path", dest="path", type=str, default='./data/coco/annotations/instances_train2017.json',
                    help="annotation json file path")
parser.add_argument("-t", "--target", dest="target", type=str, default='./UNAset',
                    help="target path")
parser.add_argument("-o", "--output", dest="output", type=str, default='UNA',
                    help="output name prefix")
parser.add_argument("-r", "--ratio", dest="ratio", type=float, default=0.1,
                    help="noise ratio")
parser.add_argument("-c", "--class_type", dest="type", type=str, default='coco', choices=["coco","voc"],
                    help="class type")
args = parser.parse_args()

path = args.path
target_path = args.target
output_file = args.output
noise_ratio = args.ratio
class_type = args.type

output_file = f"{output_file}{int(noise_ratio*100)}.json"
os.makedirs(args.target, exist_ok=True)
save_path = os.path.join(args.target, output_file)

with open(path,"r") as json_file:
    original_anns = json.load(json_file)

anns = copy.deepcopy(original_anns)
ann_len = len(anns['annotations'])

print(f"noise ratio : {noise_ratio} || class type : {class_type}")

"""### noise flag ###"""
for ann in anns['annotations']:
    ann['n_loc'] = False; ann['n_clf'] = False; ann['n_bogus'] = False

"""### missing anno ###"""
def make_missing(anns, noise_ratio, ann_len, random_seed=0):
    random.seed(random_seed)
    num_noise = int(ann_len * noise_ratio) 

    idx = list(range(ann_len))
    random.shuffle(idx)
    noise_idx = idx[:num_noise]
    noise_idx = sorted(noise_idx,reverse=True)

    for index in noise_idx:
        del anns['annotations'][index]
    
    return anns
        
"""### localization ###"""
def make_loc(anns, noise_ratio, ann_len, random_seed=1):
    random.seed(random_seed)
    image_ids_annotations = defaultdict(list)

    for ann in anns['annotations']:
        image_id = ann['image_id']
        image_ids_annotations[image_id].append(ann)
        
    image_ids_to_wh = defaultdict(list)

    for ann in anns['images']:
        image_ids_to_wh[ann['id']].append(ann['width'])
        image_ids_to_wh[ann['id']].append(ann['height'])
        
    data_len = len(anns["annotations"])
    num_noise = int(noise_ratio*ann_len)

    idx = list(range(data_len))
    random.shuffle(idx)
    noise_idx = idx[:num_noise]

    for i in noise_idx:
        x,y,w,h = anns['annotations'][i]['bbox']
        img_w,img_h = image_ids_to_wh[anns["annotations"][i]['image_id']]
        
        xlist = sorted([uniform(x-w/2,x+(3/2)*w),uniform(x-w/2,x+(3/2)*w)])
        ylist = sorted([uniform(y-h/2,y+(3/2)*h),uniform(y-h/2,y+(3/2)*h)])
        xlist = list(map(lambda x: min(max(0,round(x,2)),img_w),xlist))
        ylist = list(map(lambda y: min(max(0,round(y,2)),img_h),ylist))
        
        new_x = xlist[0]
        new_y = ylist[0]
        new_w = round(xlist[1]-xlist[0],2)
        new_h = round(ylist[1]-ylist[0],2)
        
        if new_w < 1e-05 or new_h < 1e-05:
            new_x = x+w/4
            new_y = y+h/4
            new_w = w/2
            new_h = h/2

        anns["annotations"][i]['bbox'] = [new_x, new_y, new_w, new_h]
        anns["annotations"][i]['n_loc'] = True
    
    return anns
    
"""### class ###"""
def make_class(anns, noise_ratio, ann_len, random_seed=2):
    random.seed(random_seed)
    data_len = len(anns["annotations"])
    # data_len = ann_len
    num_noise = int(noise_ratio*ann_len)
    num_category = max([anns["annotations"][i]["category_id"] for i in range(data_len)])

    idx = list(range(data_len))
    random.shuffle(idx)
    noise_idx = idx[:num_noise]

    for i in noise_idx:
        anns["annotations"][i]["category_id"] = random.choice(class_type)
        anns["annotations"][i]["n_clf"] = True
        
    return anns
    
"""### bogus ###"""
def make_bogus(anns,noise_ratio, ann_len, random_seed=3):
    random.seed(random_seed)
    num_of_images = len(anns['images'])
    max_id_plus_one = max([ann['id'] for ann in anns["annotations"]]) + 1

    image_idx = 0
    for i in range(round(ann_len * noise_ratio / (1-noise_ratio))): 
        image_idx = i % num_of_images
        image_h = anns['images'][image_idx]['height']
        image_w = anns['images'][image_idx]['width']
        
        xlist = sorted([uniform(0,image_w),uniform(0,image_w)])
        ylist = sorted([uniform(0,image_h),uniform(0,image_h)])
        xlist = list(map(lambda x: round(x,2),xlist))
        ylist = list(map(lambda y: round(y,2),ylist))
        
        new_x = xlist[0]
        new_y = ylist[0]
        new_w = round(xlist[1]-xlist[0],2)
        new_h = round(ylist[1]-ylist[0],2)
        
        area = new_w * new_h
        iscrowd = 0
        image_id = anns['images'][image_idx]['id']
        bbox = [new_x, new_y, new_w, new_h]
        category_id = random.choice(class_type)
        id = max_id_plus_one + i 
        
        anns['annotations'].append({'area': area, 'iscrowd': iscrowd, 'image_id': image_id, 'bbox': bbox, 'category_id': category_id, 'id': id, 'n_loc': True, 'n_clf': True, 'n_bogus': True})

    return anns

"""### SAVE ###"""
if __name__ == '__main__':
    anns = make_missing(anns, noise_ratio, ann_len, random_seed=0)
    anns = make_loc(anns, noise_ratio, ann_len, random_seed=1)
    anns = make_class(anns, noise_ratio, ann_len, random_seed=2)
    anns = make_bogus(anns,noise_ratio, ann_len, random_seed=3)

with open(save_path,"w") as outfile:
    json.dump(anns, outfile, indent=4)

print("done")
