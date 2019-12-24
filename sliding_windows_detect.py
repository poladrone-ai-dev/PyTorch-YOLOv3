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

def calculate_iou(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA["x1"], boxB["x1"])
    yA = max(boxA["y1"], boxB["y1"])
    xB = min(boxA["x2"], boxB["x2"])
    yB = min(boxA["y2"], boxB["y2"])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth rectangles
    boxAArea = (boxA["x2"] - boxA["x1"] + 1) * (boxA["y2"] - boxA["y1"] + 1)
    boxBArea = (boxB["x2"] - boxB["x1"] + 1) * (boxB["y2"] - boxB["y1"] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou

# def detect_image(img, model, small_image=False):
#     img_size = 416 # don't change this
#     conf_thres = 0.8
#     nms_thres = 0.4
#
#     # scale and pad image
#     ratio = min(img_size/img.shape[0], img_size/img.shape[1])
#
#     if not small_image:
#         imw = round(img.shape[0] * ratio)
#         imh = round(img.shape[1] * ratio)
#     else:
#         imw = img.shape[0]
#         imh = img.shape[1]
#
#     img_transforms = transforms.Compose([transforms.Resize((imh, imw)),
#          transforms.Pad((max(int((imh-imw) / 2), 0), max(int((imw-imh) / 2), 0), max(int((imh-imw) / 2), 0), \
#                          max(int((imw-imh) / 2), 0)), (128, 128, 128)), transforms.ToTensor(), ])
#
#     resize_transform = transforms.Compose([transforms.Resize(imh, imw), transforms.ToTensor(), ])
#
#     # convert PIL image to Tensor
#     img = Image.fromarray(img)
#     if not small_image:
#         image_tensor = img_transforms(img).float()
#         image_tensor = image_tensor.unsqueeze_(0)
#     else:
#         image_tensor = resize_transform(img).float()
#         image_tensor = image_tensor.unsqueeze_(0)
#
#     input_img = Variable(image_tensor.type(Tensor))
#
#     # run inference on the model and get detections
#     with torch.no_grad():
#         detections = model(input_img)
#         detections = non_max_suppression(detections, conf_thres, nms_thres)
#
#     return detections[0]

def detect_image(img, model):
    img_size = 416 # don't change this
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

    ### WAN added @ 17_12_2019
    parser.add_argument("--model_def", type=str, default="config/yolov3-custom.cfg", help="path to model definition file")

    # parser.add_argument("--model_def", type=str, default="WAN_Yolov3/yolov3.cfg",help="path to model definition file")

    # parser.add_argument("--weights_path", type=str, default="checkpoints_latest_3rd_Dec_2019/yolov3_ckpt_251.pth",help="path to weights file")
    parser.add_argument("--weights_path", type=str, default="pth_to_onnx/yolov3_ckpt_0.weights",
                        help="path to weights file")

    ### WAN added @ 17_12_2019
    #parser.add_argument("--weights_path", type=str, default="checkpoints_latest_3rd_Dec_2019/newYolov3.weights",help="path to weights file")
    #parser.add_argument("--weights_path", type=str, default="WAN_Yolov3/yolov3.weights",help="path to weights file")

    ### WAN added @ 17_12_2019
    parser.add_argument("--class_path", type=str, default="data/coco.names", help="path to class label file")

    #parser.add_argument("--class_path", type=str, default="WAN_Yolov3/coco.names", help="path to class label file")

    parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.4, help="iou threshold for non-maximum suppression")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--img_size", type=int, default=500, help="size of each image dimension")
    parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
    parser.add_argument("--checkpoint_model", type=str, help="path to checkpoint model")
    parser.add_argument("--x_stride", type=int, default=200, help="width stride of the sliding window in pixels")
    parser.add_argument("--y_stride", type=int, default=200, help="height stride of the sliding window in pixels")

    ### WAN added @ 17_12_2019
    parser.add_argument("--image", type=str, default='output/03.jpg',
                        help="the image to apply sliding windows on")

    ### WAN added @ 17_12_2019
    #parser.add_argument("--image", type=str, default='output/sample11.jpg',help="the image to apply sliding windows on")

    opt = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Darknet(opt.model_def, img_size=opt.img_size).to(device)

    if opt.weights_path.endswith(".weights"):
        # Load darknet weights
        print("Loaded the full weights with network architecture.")
        model.load_darknet_weights(opt.weights_path)
    else:
        # Load checkpoint weights
        print("Loaded only the trained weights.")
        model.load_state_dict(torch.load(opt.weights_path))

    model.eval() # Set in evaluation mode

    classes = load_classes(opt.class_path)  # Extracts class labels from file
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    im = Image.open(opt.image)
    image_width, image_height = im.size

    small_image = False

    if image_width <= 500 and image_height <= 500:
        [winW, winH] = [250, 250]
        opt.x_stride = int(winW / 2)
        opt.y_stride = int(winH / 2)
        small_image = True

    elif image_width > 500 and image_width <= 1000 and image_height > 500 and image_height <= 1000:
        pass

    else:
        [winW, winH] = [500, 500]
        opt.x_stride = int(winW / 2)
        opt.y_stride = int(winH / 2)

    image = cv2.imread(opt.image)

    cv2.namedWindow("output", cv2.WINDOW_NORMAL)
    bbox_color = 'red'
    output_path = r'D:\PyTorch-YOLOv3-master\sliding_windows_output'
    window_idx = 0

    output_json = {}

    if os.path.exists(os.path.join(output_path, "Results_brian.txt")):
        os.remove(os.path.join(output_path, "Results_brian.txt"))

    fp = open(os.path.join(output_path, "Results_brian.txt"), "a")

    for (x, y, window) in sliding_window(image, x_stepSize=opt.x_stride, y_stepSize=opt.y_stride,
                                         windowSize=[winW, winH]):
        # THIS IS WHERE YOU WOULD PROCESS YOUR WINDOW, SUCH AS APPLYING A
        # MACHINE LEARNING CLASSIFIER TO CLASSIFY THE CONTENTS OF THE
        # WINDOW

        window_name = "window" + str(window_idx)
        if window_name not in output_json:
            output_json[window_name] = {}

        output_json[window_name]["tile"] = [x / image_width, y / image_height]
        detections = detect_image(window, model)

        if detections is not None:
            detections = rescale_boxes(detections, opt.img_size, window.shape[:2])
            unique_labels = detections[:, -1].cpu().unique()
            n_cls_preds = len(unique_labels)
            box_idx = 0

            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                if x1 < 0:
                    x1 = torch.tensor(0)

                if y1 < 0:
                    y1 = torch.tensor(0)

                if x2 > image_width:
                    x2 = torch.tensor(image_width)

                if y2 > image_height:
                    y2 = torch.tensor(image_height)

                box_w = x2 - x1
                box_h = y2 - y1

                if classes[int(cls_pred)] != "palm0":
                    box_name = "box" + str(box_idx)
                    if box_name not in output_json[window_name]:
                        output_json[window_name][box_name] = \
                        {
                            "x1": round(x1.item(), 3),
                            "y1": round(y1.item(), 3),
                            "x2": round(x2.item(), 3),
                            "y2": round(y2.item(), 3),
                            "width": round(box_w.item(), 3),
                            "height": round(box_h.item(), 3),
                            "conf": round(conf.item(), 3),
                            "cls_conf": round(cls_conf.data.tolist(), 3),
                            "cls_pred": classes[int(cls_pred)]
                        }

                    if window_idx - 1 >= 0:
                        prev_window_name = "window" + str(window_idx - 1)
                        for prev_box in output_json[prev_window_name]:
                            if "box" in prev_box:
                                if prev_box != '{}':
                                    iou = calculate_iou(output_json[window_name][box_name],
                                                        output_json[prev_window_name][prev_box])
                                    if iou < 0.4:
                                        cv2.rectangle(image, (x + x1, y + y1), (x + x2, y + y2), (0, 0, 255), 2)
                                        cv2.putText(image, classes[int(cls_pred)], (int(x + x1), int(y + y1)), \
                                                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), lineType=cv2.LINE_AA)

                                        conf = round(cls_conf.data.tolist(), 2)
                                        fp.write(str(classes[int(cls_pred)]) + " " + str(conf).replace(str(conf)[0], '')
                                                 + " " + str(int(x1.item())) + " " + str(int(y1.item())) + " "
                                                 + str(int(box_w.item())) + " " + str(int(box_h.item())) + '\n')
                                        box_idx += 1

                    # cv2.rectangle(image, (x + x1, y + y1), (x + x2, y + y2), (0, 0, 255), 2)
                    # cv2.putText(image, classes[int(cls_pred)], (int(x + x1), int(y + y1)), \
                    #             cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), lineType=cv2.LINE_AA)
                    # conf = round(cls_conf.data.tolist(), 2)
                    # fp.write(str(classes[int(cls_pred)]) + " " + str(conf).replace(str(conf)[0], '')
                    #          + " " + str(int(x1.item())) + " " + str(int(y1.item())) + " "
                    #          + str(int(box_w.item())) + " " + str(int(box_h.item())) + '\n')
                    # box_idx += 1

        # cv2.rectangle(image, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
        cv2.imshow("Window", image)

        # cv2.imwrite(os.path.join(output_path, "picture_" + str(box_idx) + ".jpeg"), image)
        window_idx += 1
        # image = cv2.imread(opt.image)
        cv2.waitKey(1)
        # time.sleep(0.1)

    fp.close()
    cv2.imwrite(os.path.join(output_path, "combined.jpeg"), image)
    with open(os.path.join(output_path, "detection.json"), "w") as img_json:
        json.dump(output_json, img_json, indent=4)


