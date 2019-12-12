from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *

import os
import sys
import time
import datetime
import argparse
import json
import re
import cv2

from PIL import Image

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator
from pyimagesearch.helpers import sliding_window

def detect_image(img, model):

    img_size = 416
    conf_thres = 0.8
    nms_thres = 0.4

    # scale and pad image
    ratio = min(img_size/img.shape[0], img_size/img.shape[1])
    imw = round(img.shape[0] * ratio)
    imh = round(img.shape[1] * ratio)
    img_transforms = transforms.Compose([transforms.Resize((imh, imw)),
         transforms.Pad((max(int((imh-imw) / 2), 0), max(int((imw-imh) / 2), 0), max(int((imh-imw) / 2), 0), \
                         max(int((imw-imh) / 2), 0)), (128, 128, 128)), transforms.ToTensor(), ])

    # convert PIL image to Tensor
    img = Image.fromarray(img)
    image_tensor = img_transforms(img).float()
    image_tensor = image_tensor.unsqueeze_(0)
    input_img = Variable(image_tensor.type(Tensor))

    # run inference on the model and get detections
    with torch.no_grad():
        detections = model(input_img)
        detections = non_max_suppression(detections, conf_thres, nms_thres)

    return detections[0]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str, default="data/samples", help="path to dataset")
    parser.add_argument("--model_def", type=str, default="config/yolov3-custom.cfg",
                        help="path to model definition file")
    parser.add_argument("--weights_path", type=str, default="checkpoints_latest_3rd_Dec_2019/yolov3_ckpt_251.pth",
                        help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/coco.names", help="path to class label file")
    parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.4, help="iou threshold for non-maximum suppression")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--img_size", type=int, default=500, help="size of each image dimension")
    parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
    parser.add_argument("--checkpoint_model", type=str, help="path to checkpoint model")
    parser.add_argument("--stride", type=int, default=50, help="stride of the sliding window in pixels")
    parser.add_argument("--image", type=str, default='output/tile_merged.jpg',
                        help="the image to apply sliding windows on")
    opt = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set up model
    model = Darknet(opt.model_def, img_size=opt.img_size).to(device)

    if opt.weights_path.endswith(".weights"):
        # Load darknet weights
        model.load_darknet_weights(opt.weights_path)
    else:
        # Load checkpoint weights
        model.load_state_dict(torch.load(opt.weights_path))

    model.eval() # Set in evaluation mode

    classes = load_classes(opt.class_path)  # Extracts class labels from file
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    im = Image.open(opt.image)
    image_width, image_height = im.size
    (winW, winH) = (500, 500)
    image = cv2.imread(opt.image)

    cv2.namedWindow("output", cv2.WINDOW_NORMAL)
    image_resized = cv2.resize(image, (1280, 1024))

    bbox_color = 'red'
    output_path = r'D:\PyTorch-YOLOv3-master\sliding_windows_output'
    box_idx = 0

    for (x, y, window) in sliding_window(image_resized, stepSize=opt.stride, windowSize=(winW, winH)):

        if window.shape[0] != winH or window.shape[1] != winW:
            continue

        # THIS IS WHERE YOU WOULD PROCESS YOUR WINDOW, SUCH AS APPLYING A
        # MACHINE LEARNING CLASSIFIER TO CLASSIFY THE CONTENTS OF THE
        # WINDOW

        detections = detect_image(window, model)
        if detections is not None:
            detections = rescale_boxes(detections, opt.img_size, window.shape[:2])
            unique_labels = detections[:, -1].cpu().unique()
            n_cls_preds = len(unique_labels)

            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                if x1 < 0:
                    x1 = torch.tensor(0)

                if y1 < 0:
                    y1 = torch.tensor(0)

                if x2 > image_width:
                    x2 = torch.tensor(image_width)

                if y2 > image_height:
                    y2 = torch.tensor(image_height)

                # print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
                # print("x1: " + str(x1))
                # print("y1: " + str(y1))
                # print("x2: " + str(x2))
                # print("y2: " + str(y2))
                # print("conf: " + str(conf))
                # print("cls_conf: " + str(cls_conf))
                # print("cls_pred: " + str(cls_pred))
                # print("\t+ Label: %s, Conf: %.5f" % (classes[int(cls_pred)], cls_conf.item()))
                # print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")

                cv2.rectangle(image_resized, (x + x1, y + y1), (x + x2, y + y2), (0, 0, 255), 2)

        cv2.rectangle(image_resized, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
        cv2.imshow("Window", image_resized)

        cv2.imwrite(os.path.join(output_path, "picture_" + str(box_idx) + ".jpeg"), image_resized)
        box_idx += 1
        image = cv2.imread(opt.image)
        image_resized = cv2.resize(image, (1280, 1024))
        cv2.waitKey(1)
        time.sleep(0.1)

