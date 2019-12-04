from __future__ import division

from models import *
from utils.logger import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *
from test import evaluate
from utils.WarmUp import GradualWarmupScheduler

from terminaltables import AsciiTable

from datetime import datetime

import os
import sys
import time
import datetime
import argparse
import wandb

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def step_decay(initial_lr):
    initial_lrate = initial_lr
    drop = 0.5
    epochs_drop = 1
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch) / epochs_drop))
    return lrate

def lambda_rule(epoch):
    lr_l = 0.2 ** epoch
    return lr_l

if __name__ == "__main__":

    with open("runtime.txt", 'w') as fp:
        fp.write("Start time: " + str(datetime.datetime.now()))

    wandb.init()
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1000, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=9, help="size of each image batch")
    parser.add_argument("--gradient_accumulations", type=int, default=2, help="number of gradient accums before step")
    parser.add_argument("--model_def", type=str, default="config/yolov3-custom.cfg", help="path to model definition file")
    parser.add_argument("--data_config", type=str, default="config/coco.data", help="path to data config file")
    parser.add_argument("--pretrained_weights", type=str, help="if specified starts from checkpoint model")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between saving model weights")
    parser.add_argument("--evaluation_interval", type=int, default=1, help="interval evaluations on validation set")
    parser.add_argument("--compute_map", default=False, help="if True computes mAP every tenth batch")
    parser.add_argument("--multiscale_training", default=True, help="allow for multi-scale training")
    opt = parser.parse_args() # command line arguments
    print("Default arguments: " + str(opt))

    # variables for keeping track of early stopping
    early_stopping_intervals = 10000
    best_loss = float("inf")
    stopping_step = 0
    global_steps = 0
    tolerance_level = 0.01
    patience = 0 # number of no improvements noted before early stopping is triggered\
    MAX_PATIENCE = 10

    logger = Logger("logs")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    os.makedirs("output", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    # Get data configuration
    data_config = parse_data_config(opt.data_config)
    train_path = data_config["train"]
    valid_path = data_config["valid"]
    class_names = load_classes(data_config["names"])

    # Initiate model
    model = Darknet(opt.model_def).to(device)
    model.apply(weights_init_normal)

    # If specified we start from checkpoint
    if opt.pretrained_weights:
        if opt.pretrained_weights.endswith(".pth"):
            model.load_state_dict(torch.load(opt.pretrained_weights))
        else:
            model.load_darknet_weights(opt.pretrained_weights)

    # Get dataloader
    dataset = ListDataset(train_path, augment=True, multiscale=opt.multiscale_training)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        # batch_size= int(int(model.hyperparams['batch']) / int(model.hyperparams['subdivisions'])),
        shuffle=True,
        num_workers=opt.n_cpu,
        pin_memory=True,
        collate_fn=dataset.collate_fn,
    )

    optimizer = torch.optim.Adam(model.parameters(),
         lr=float(model.hyperparams['learning_rate']),
         weight_decay=float(model.hyperparams['decay'])
    )

    initial_learning_rate = model.hyperparams['learning_rate']
    initial_decay = model.hyperparams['decay']

    # scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.epochs)
    # lambda1 = lambda epoch: 1
    # lambda2 = lambda epoch: 0.95 * 1
    # scheduler_lambda = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    # scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=8, total_epoch=2, after_scheduler=scheduler_lambda)

    metrics = [
        "grid_size",
        "loss",
        "x",
        "y",
        "w",
        "h",
        "conf",
        "cls",
        "cls_acc",
        "recall50",
        "recall75",
        "precision",
        "conf_obj",
        "conf_noobj",
        "F1_score",
        "iou50",
        "iou75"
    ]

    for epoch in range(opt.epochs):
        # scheduler_warmup.step()
        model.train()
        start_time = time.time()
        for batch_i, (_, imgs, targets) in enumerate(dataloader):
            batches_done = len(dataloader) * epoch + batch_i

            imgs = Variable(imgs.to(device))
            targets = Variable(targets.to(device), requires_grad=False)

            loss, outputs = model(imgs, targets)
            loss.backward()

            if batches_done % opt.gradient_accumulations:
                # Accumulates gradient before each step
                optimizer.step()
                optimizer.zero_grad()
                wandb.log({"learning_rate": get_lr(optimizer)})

            # ----------------
            #   Log progress
            # ----------------

            log_str = "\n---- [Epoch %d/%d, Batch %d/%d] ----\n" % (epoch, opt.epochs, batch_i, len(dataloader))

            metric_table = [["Metrics", *[f"YOLO Layer {i}" for i in range(len(model.yolo_layers))]]]

            # Log metrics at each YOLO layer
            for i, metric in enumerate(metrics):
                formats = {m: "%.6f" for m in metrics}
                formats["grid_size"] = "%2d"
                formats["cls_acc"] = "%.2f%%"
                row_metrics = [formats[metric] % yolo.metrics.get(metric, 0) for yolo in model.yolo_layers]
                metric_table += [[metric, *row_metrics]]

                if metric == "cls_acc":
                    wandb.log({"cls_acc": yolo.metrics.get(metric, 0)})

                if metric == "conf":
                    wandb.log({"conf": yolo.metrics.get(metric, 0)})

                if metric == "conf_obj":
                    wandb.log({"conf_obj": yolo.metrics.get(metric, 0)})

                if metric == "conf_noobj":
                    wandb.log({"conf_noobj": yolo.metrics.get(metric, 0)})

                if metric == "precision":
                    wandb.log({"precision": yolo.metrics.get(metric, 0)})

                if metric == "recall50":
                    wandb.log({"recall50": yolo.metrics.get(metric, 0)})

                if metric == "recall75":
                    wandb.log({"recall75": yolo.metrics.get(metric, 0)})

                if metric == "F1_score":
                    wandb.log({"F1_score": yolo.metrics.get(metric, 0)})

                if metric == "iou50":
                    wandb.log({"IOU50": yolo.metrics.get(metric, 0)})

                if metric == "iou75":
                    wandb.log({"IOU75": yolo.metrics.get(metric, 0)})

                # wandb.log({"learning_rate": get_lr(optimizer)})

                # Tensorboard logging
                tensorboard_log = []
                for j, yolo in enumerate(model.yolo_layers):
                    for name, metric in yolo.metrics.items():
                        if name != "grid_size":
                            tensorboard_log += [(f"{name}_{j+1}", metric)]
                tensorboard_log += [("loss", loss.item())]
                logger.list_of_scalars_summary(tensorboard_log, batches_done)

            log_str += AsciiTable(metric_table).table
            log_str += f"\nTotal loss {loss.item()}"

            wandb.log({"total_loss": loss.item()})

            # Determine approximate time left for epoch
            epoch_batches_left = len(dataloader) - (batch_i + 1)
            time_left = datetime.timedelta(seconds=epoch_batches_left * (time.time() - start_time) / (batch_i + 1))
            log_str += f"\n---- ETA {time_left}"

            print(log_str)

            model.seen += imgs.size(0)

            if global_steps % early_stopping_intervals == 0:
                print(str(global_steps) + " steps mark reached.")
                print("Current Epoch: " + str(epoch))
                print("Current Step: " + str(global_steps))
                print("Best loss recorded: " + str(best_loss))
                print("Current loss: " + str(loss.item()))

                if (abs(loss.item() - best_loss) / loss.item()) > tolerance_level:
                    print("Updating best loss to current loss...")
                    best_loss = loss.item()
                else:
                    patience += 1
                    if patience >= MAX_PATIENCE:
                        print("Early stopping triggered.")
                        print("Total loss when early stopping is triggered: " + str(loss.item()) + ".")
                        break

            global_steps += 1

        if epoch % opt.evaluation_interval == 0:
            print("\n---- Evaluating Model ----")
            # Evaluate the model on the validation set
            precision, recall, AP, f1, ap_class = evaluate(
                model,
                path=valid_path,
                iou_thres=0.5,
                conf_thres=0.5,
                nms_thres=0.5,
                img_size=opt.img_size,
                batch_size=8,
            )
            evaluation_metrics = [
                ("val_precision", precision.mean()),
                ("val_recall", recall.mean()),
                ("val_mAP", AP.mean()),
                ("val_f1", f1.mean()),
            ]
            logger.list_of_scalars_summary(evaluation_metrics, epoch)

            # Print class APs and mAP
            ap_table = [["Index", "Class name", "AP"]]
            for i, c in enumerate(ap_class):
                ap_table += [[c, class_names[c], "%.5f" % AP[i]]]
            print(AsciiTable(ap_table).table)
            print(f"---- mAP {AP.mean()}")

        if epoch % opt.checkpoint_interval == 0:
            torch.save(model.state_dict(), f"checkpoints/yolov3_ckpt_%d.pth" % epoch)


    with open("runtime.txt", 'a') as fp:
        fp.write("Finish time: " + str(datetime.datetime.now()))