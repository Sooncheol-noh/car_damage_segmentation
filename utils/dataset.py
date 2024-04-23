import os
from torchvision.transforms import ColorJitter
import numpy as np
from datasets import Dataset, Image


def create_dataset(image_root, mask_png_root, file_name_list, key_list):
    image_path_list = [os.path.join(image_root, file_name + '.jpg') for file_name in file_name_list]
    dataset_dict = {}
    for key in key_list:
        dataset_dict[key]=[os.path.join(mask_png_root, file_name + f'_{key}.png') for file_name in file_name_list]
    dataset_dict['image']=image_path_list

    dataset = Dataset.from_dict(dataset_dict)
    dataset = dataset.cast_column("image", Image())
    for key in key_list:
        dataset = dataset.cast_column(key, Image())

    return dataset

class CustomTransform():
    def __init__(self, image_processor, key_list, jitter = None):
        self.image_processor = image_processor
        self.key_list = key_list
        self.jitter = jitter

    def __call__(self, example_batch):
        if self.jitter is None:
            images = [x.convert('RGB') for x in example_batch["image"]]
        else:
            images = [self.jitter(x.convert('RGB')) for x in example_batch["image"]]
        labels = [{key: example_batch[key][i] for key in self.key_list} for i in range(len(images))]
        inputs = self.image_processor(images, labels)
        inputs['labels'] = [np.array([label[key] for key in self.key_list]) for label in inputs['labels']]
        return inputs