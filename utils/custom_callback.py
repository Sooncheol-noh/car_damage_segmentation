import matplotlib.pyplot as plt
from transformers import TrainerCallback
import torch.nn as nn
import numpy as np
from utils.metric import BinaryMetrics
from utils.etc import minmax_normalize

binary_metrics = BinaryMetrics()

class ImageLoggingCallback(TrainerCallback):
    def __init__(self, tb_writer, **kwargs):
        super(ImageLoggingCallback, self).__init__(**kwargs)
        self.tb_writer = tb_writer
    
    def on_evaluate(self, args, state, control, model, eval_dataloader, **kwargs): 
        if state.global_step % 50 != 0:
            return

        key_list = ["Scratched", "Breakage", "Separated", "Crushed", "Total"]
        sigmoid = nn.Sigmoid()
        for idx, data in enumerate(eval_dataloader):
            if idx == 0:
                pixel_values = data['pixel_values'][0]
                labels = data['labels'][0]
            elif idx == 1:
                pixel_values = data['pixel_values'][0]
                labels = data['labels'][0]
            else:
                break
                        
            outputs = model(pixel_values[None])
            logits = outputs.get("logits")
            upsampled_logits = nn.functional.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
            _, _, _, _, _, iou, f1 = binary_metrics(labels[None], upsampled_logits)
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
            self.tb_writer.add_figure(f'prediction_example_test{idx}', fig, global_step=state.global_step)
