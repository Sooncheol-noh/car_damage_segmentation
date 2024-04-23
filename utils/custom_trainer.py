from transformers import Trainer
import torch
import torch.nn as nn

class CustomTrainer(Trainer):
    def __init__(self, loss_function, **kwargs):
        super(CustomTrainer, self).__init__(**kwargs)
        self.loss_function = loss_function

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop('labels')
        outputs = model(**inputs)
        logits = outputs.get("logits")
        upsampled_logits = nn.functional.interpolate(
            logits, size=labels.shape[-2:], mode="bilinear", align_corners=False
        )
        
        # bceloss = nn.BCEWithLogitsLoss(reduction="none")
        # loss = bceloss(upsampled_logits, labels.float()).mean()

        # diceloss = DiceLoss()
        # loss = diceloss(upsampled_logits, labels.float())
        loss = self.loss_function(upsampled_logits, labels.float())
        loss = torch.mean(loss)
        return (loss, outputs) if return_outputs else loss
