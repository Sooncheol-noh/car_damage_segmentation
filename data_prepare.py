import random
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import skimage
from tqdm import tqdm

image_root = "/home/colin/mnt_nas06/car_damage_detect/car_damage_dataset/data/Training/raw_data/damage"
label_root = "/home/colin/mnt_nas06/car_damage_detect/car_damage_dataset/data/Training/labeling/damage"

image_list = os.listdir(image_root)
label_list = os.listdir(label_root)

id2label = {int(k): v for k, v in enumerate(['Scratched', 'Breakage', 'Separated', 'Crushed'])}
label2id = {v: k for k, v in id2label.items()}

print('image list: ', len(image_list))
print('label list: ', len(label_list))

temp_list = list(set([l.replace('.json', '') for l in label_list]) & set([i.replace('.jpg', '') for i in image_list]))

print('temp_list: ', len(temp_list))

destination_root = "/home/colin/Projects/car_damage_segmentation/dataset/"

pre_list = os.listdir(os.path.join(destination_root, 'images'))

for file_name in tqdm(temp_list):
    if file_name + '.jpg' not in pre_list:
        t = 1
    else:
        continue

    with open(os.path.join(label_root, f"{file_name}.json"), "r") as f:
        label = json.load(f)

    image = Image.open(os.path.join(image_root, f"{file_name}.jpg"))
    try:
        image.save(os.path.join(destination_root, "images", f"{file_name}.jpg"))
    except:
        image.convert('RGB').save(os.path.join(destination_root, "images", f"{file_name}.jpg"))

    mask_array = np.zeros((4, image.size[1], image.size[0]), dtype = np.bool_)
    for annotation in label['annotations']:
        polygon = np.array(annotation['segmentation'][0][0])
        try:
            idx = label2id[annotation['damage']]
            mask_array[idx] += skimage.draw.polygon2mask(image.size[::-1], polygon[:, ::-1])
        except:
            continue
        if np.sum(mask_array) == 0:
            print(file_name)

    for idx, mask in enumerate(mask_array):
        im = Image.fromarray(mask)
        im.save(os.path.join(destination_root, "annotations", f"{file_name}_{id2label[idx]}.png"))