import os
import torch
import numpy as np
from tqdm import tqdm
import pickle
import torch.nn as nn
import matplotlib.pyplot as plt
from transformers import AutoModelForSemanticSegmentation
from transformers import AutoConfig

from utils.custom_image_porcessor import SegformerMultiLabelImageProcessor
from utils.dataset import CustomTransform, create_dataset
from utils.metric import BinaryMetrics
from utils.etc import minmax_normalize

from utils.metric import BinaryMetrics
import argparse

def parsing_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight_path", default="./logs/try0/checkpoint-116000", type=str)
    parser.add_argument("--saving_path", default="./results", type=str)
    parser.add_argument("--data_root", default="./dataset", type=str)
    parser.add_argument("--test_list", default="test_fold1.txt", type=str)
    parser.add_argument(
        "--key_list",
        default=["Scratched", "Breakage", "Separated", "Crushed"],
        nargs="+",
        type=str,
    )
    return parser.parse_args()

def main():
    args = parsing_argument()

    with open(os.path.join(args.data_root, args.test_list), "r") as f:
        test_list = f.readlines()
        test_list = [name.strip() for name in test_list]

    key_list = args.key_list
    checkpoint = "nvidia/mit-b0"
    image_processor = SegformerMultiLabelImageProcessor.from_pretrained(
            checkpoint, do_reduce_labels=False, image_std=[1.0, 1.0, 1.0]
        )

    test_image_root = os.path.join(args.data_root, "test", "images")
    test_mask_png_root = os.path.join(args.data_root, "test", "annotations")
    val_transforms = CustomTransform(image_processor, key_list, None)

    test_dataset = create_dataset(
        test_image_root, test_mask_png_root, test_list, key_list
    )
    test_dataset.set_transform(val_transforms)

    weight_path = args.weight_path
    results_saving_path = os.path.join(args.saving_path, weight_path.split('/')[-1], 'prediction_image')
    os.makedirs(results_saving_path, exist_ok = True)
    model = AutoModelForSemanticSegmentation.from_pretrained(weight_path)
    device = torch.device('cuda')
    model.to(device)
    model.eval()

    binary_metrics = BinaryMetrics()

    key_list = args.key_list + ["Total"]
    sigmoid = nn.Sigmoid()
    iou_list = []
    f1_list = []

    for idx in tqdm(range(len(test_dataset))):
        with torch.no_grad():
            data = test_dataset[idx]
            pixel_values = torch.from_numpy(data['pixel_values']).to(device)
            labels = torch.from_numpy(data['labels']).to(device)
                        
            outputs = model(pixel_values[None])
            logits = outputs.get("logits")
            upsampled_logits = nn.functional.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
            _, _, _, _, _, iou, f1 = binary_metrics(labels[None], upsampled_logits)
            iou_list.append(iou[0].tolist())
            f1_list.append(f1[0].tolist())
            pixel_values = pixel_values.cpu().numpy()
            fig, axes = plt.subplots(2, 6, figsize = (15, 5))
            axes[0][5].axis('off')
            axes[1][5].axis('off')
            axes[0][5].imshow(minmax_normalize(pixel_values).transpose(1,2,0))

            pred_mask = np.array(sigmoid(upsampled_logits.detach().cpu()) > 0.5)
            pred_mask = np.concatenate([pred_mask[0], np.sum(pred_mask, axis = 1).astype(bool)])

            gt_mask = labels.cpu().numpy()
            gt_mask = np.concatenate([gt_mask, np.sum(gt_mask, axis = 0)[None]]).astype(bool)

            for i in range(len(key_list)):
                axes[1][i].axis('off')
                axes[1][i].set_title(f"F1 : {round(f1[0][i].item(), 3)} / IoU : {round(iou[0][i].item(), 3)}")
                mask = pred_mask[i][None]
                mask = np.concatenate([mask, np.zeros_like(mask), np.zeros_like(mask)])
                axes[1][i].imshow(minmax_normalize(minmax_normalize(pixel_values)*0.7 + mask*0.3).transpose(1,2,0))

            for i in range(len(key_list)):
                axes[0][i].axis('off')
                axes[0][i].set_title(f"{key_list[i]}")
                mask = gt_mask[i][None]
                mask = np.concatenate([mask, np.zeros_like(mask), np.zeros_like(mask)])
                axes[0][i].imshow(minmax_normalize(minmax_normalize(pixel_values)*0.7 + mask*0.3).transpose(1,2,0))
            fig.tight_layout(rect=[0, 0, 1, 1])
            plt.close(fig)
            fig.savefig(os.path.join(results_saving_path, str(idx)+'.png'))

    with open(os.path.join(args.saving_path, weight_path.split('/')[-1], 'iou.pickle'), 'wb') as fw:
        pickle.dump(iou_list, fw, protocol=pickle.HIGHEST_PROTOCOL)

    with open(os.path.join(args.saving_path, weight_path.split('/')[-1], 'f1.pickle'), 'wb') as fw:
        pickle.dump(f1_list, fw, protocol=pickle.HIGHEST_PROTOCOL)

    print("Weight path:", weight_path)

    iou_mean = np.mean(np.array(iou_list), axis = 0)
    f1_mean = np.mean(np.array(f1_list), axis = 0)

    for i, key in enumerate(key_list):
        print(f"iou ({key}):    \t{round(iou_mean[i], 4)}")
    print("")
    for i, key in enumerate(key_list):
        print(f"f1 ({key}):     \t{round(f1_mean[i], 4)}")

if __name__ == "__main__":
    main()