import os
import argparse
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from transformers import AutoModelForSemanticSegmentation, TrainingArguments
from transformers import AutoConfig
from transformers.integrations import TensorBoardCallback

from torchvision.transforms import ColorJitter

from utils.custom_image_porcessor import SegformerMultiLabelImageProcessor
from utils.custom_loss import BinaryFocalLoss
from utils.dataset import CustomTransform, create_dataset
from utils.custom_trainer import CustomTrainer
from utils.metric import BinaryMetrics
from utils.custom_callback import ImageLoggingCallback
from utils.etc import minmax_normalize, compute_metrics

def parsing_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="./sampled_data", type=str)
    parser.add_argument("--train_list", default="train_fold_total.txt", type=str)
    parser.add_argument("--test_list", default="test_fold1.txt", type=str)
    parser.add_argument("--output_dir", default="./temp_logs", type=str)
    parser.add_argument("--tensorboard_logging_dir", default=None, type=str)
    parser.add_argument(
        "--key_list",
        default=["Scratched", "Breakage", "Separated", "Crushed"],
        nargs="+",
        type=str,
    )

    parser.add_argument("--max_epoch", default=50, type=int)
    parser.add_argument("--batch_size", default=60, type=int)
    parser.add_argument("--lr", default=5e-4, type=float)
    parser.add_argument("--weight_decay", default=0.05, type=float)
    parser.add_argument(
        "--steplr_milestones",
        default=[200, 1000, 2000, 3000],
        nargs="+",
        type=int,
    )
    

    parser.add_argument("--pretrained_checkpoint", default="nvidia/mit-b0", type=str)

    return parser.parse_args()


def main():
    args = parsing_argument()

    with open(os.path.join(args.data_root, args.train_list), "r") as f:
        train_list = f.readlines()
        train_list = [name.strip() for name in train_list]
    with open(os.path.join(args.data_root, args.test_list), "r") as f:
        test_list = f.readlines()
        test_list = [name.strip() for name in test_list]

    # train_image_root = os.path.join(args.data_root, "train", "images")
    # train_mask_png_root = os.path.join(args.data_root, "train", "annotations")
    # test_image_root = os.path.join(args.data_root, "test", "images")
    # test_mask_png_root = os.path.join(args.data_root, "test", "annotations")
    
    train_image_root = os.path.join(args.data_root, "images")
    train_mask_png_root = os.path.join(args.data_root, "annotations")
    test_image_root = os.path.join(args.data_root, "images")
    test_mask_png_root = os.path.join(args.data_root, "annotations")

    id2label = {int(k): v for k, v in enumerate(args.key_list)}
    label2id = {v: k for k, v in id2label.items()}
    num_labels = len(id2label)

    config = AutoConfig.from_pretrained(args.pretrained_checkpoint)
    config.id2label = id2label
    config.label2id = label2id

    model = AutoModelForSemanticSegmentation.from_pretrained(
        args.pretrained_checkpoint, config=config
    )
    train_layer_key_list = [
        "decode_head.batch_norm.bias",
        "decode_head.batch_norm.num_batches_tracked",
        "decode_head.batch_norm.running_mean",
        "decode_head.batch_norm.running_var",
        "decode_head.batch_norm.weight",
        "decode_head.classifier.bias",
        "decode_head.classifier.weight",
        "decode_head.linear_c.0.proj.bias",
        "decode_head.linear_c.0.proj.weight",
        "decode_head.linear_c.1.proj.bias",
        "decode_head.linear_c.1.proj.weight",
        "decode_head.linear_c.2.proj.bias",
        "decode_head.linear_c.2.proj.weight",
        "decode_head.linear_c.3.proj.bias",
        "decode_head.linear_c.3.proj.weight",
        "decode_head.linear_fuse.weight",
    ]
    for k, v in model.named_parameters():
        if k not in train_layer_key_list:
            v.requires_grad = False
        else:
            v.requires_grad = True
        if "segformer.encoder.block.3" in k:
            v.requires_grad = True

    image_processor = SegformerMultiLabelImageProcessor.from_pretrained(
        args.pretrained_checkpoint, do_reduce_labels=False, image_std=[1.0, 1.0, 1.0]
    )
    jitter = ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.1)

    train_transforms = CustomTransform(image_processor, args.key_list, jitter)
    train_dataset = create_dataset(
        train_image_root, train_mask_png_root, train_list, args.key_list
    )
    train_dataset.set_transform(train_transforms)

    val_transforms = CustomTransform(image_processor, args.key_list, None)
    test_dataset = create_dataset(
        test_image_root, test_mask_png_root, test_list[:200], args.key_list
    )
    test_dataset.set_transform(val_transforms)

    binaryfocalloss = BinaryFocalLoss(alpha=0.9, gamma=2)
    if args.tensorboard_logging_dir is None:
        args.tensorboard_logging_dir = os.path.join(args.output_dir, 'tb_log')

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        logging_dir = args.tensorboard_logging_dir,
        learning_rate=args.lr,
        num_train_epochs=args.max_epoch,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=100,
        save_total_limit=100,
        evaluation_strategy="steps",
        save_strategy="steps",
        save_steps=50,
        eval_steps=None,
        logging_strategy="steps",
        logging_steps=10,
        eval_accumulation_steps=2,
        remove_unused_columns=False,
        push_to_hub=False,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.05)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=args.steplr_milestones, gamma=0.5
    )

    trainer = CustomTrainer(
        loss_function=binaryfocalloss,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        optimizers=(optimizer, lr_scheduler),
        compute_metrics=compute_metrics,
    )

    for callback in trainer.callback_handler.callbacks:
        if isinstance(callback, TensorBoardCallback):
            callback.tb_writer = SummaryWriter(log_dir = args.tensorboard_logging_dir)
            image_logging_callback = ImageLoggingCallback(callback.tb_writer)
            break
    trainer.add_callback(image_logging_callback)
    trainer.train()

if __name__ == "__main__":
    main()
