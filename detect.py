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
import overlap_detection
import cv2

from PIL import Image

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

def find_nth(haystack, needle, n):
    start = haystack.find(needle)
    while start >= 0 and n > 1:
        start = haystack.find(needle, start + len(needle))
        n -= 1
    return start

def get_tile_coordinates(filename):
    underscore_count = len(re.findall('_', filename))
    first_underscore = find_nth(filename, '_', underscore_count - 1)
    second_underscore = find_nth(filename, '_', underscore_count)
    dot_index = find_nth(filename, '.', 1)
    x_coord = int(filename[first_underscore + 1: second_underscore])
    y_coord = int(filename[second_underscore + 1: dot_index])
    coordinates = [x_coord, y_coord]
    return coordinates

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str, default="data/samples", help="path to dataset")
    parser.add_argument("--model_def", type=str, default="config/yolov3-custom.cfg", help="path to model definition file")
    parser.add_argument("--weights_path", type=str, default="checkpoints_latest_3rd_Dec_2019/yolov3_ckpt_251.pth", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/coco.names", help="path to class label file")
    parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.4, help="iou threshold for non-maximum suppression")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--checkpoint_model", type=str, help="path to checkpoint model")
    opt = parser.parse_args()
    print(opt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs("output", exist_ok=True)

    # Set up model
    model = Darknet(opt.model_def, img_size=opt.img_size).to(device)

    if opt.weights_path.endswith(".weights"):
        # Load darknet weights
        model.load_darknet_weights(opt.weights_path)
    else:
        # Load checkpoint weights
        model.load_state_dict(torch.load(opt.weights_path))

    model.eval()  # Set in evaluation mode

    dataloader = DataLoader(
        ImageFolder(opt.image_folder, img_size=opt.img_size),
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.n_cpu,
    )

    classes = load_classes(opt.class_path)  # Extracts class labels from file

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    imgs = []  # Stores image paths
    img_detections = []  # Stores detections for each image index

    # json file containing tiff information
    # tiff_info_file = r'meraub.json'

    # tiff file infos
    upper_left_x = 0
    upper_left_y = 0
    lower_right_x = 0
    lower_right_y = 0
    width = 0
    height = 0
    xres = 0
    yres = 0

    # with open(tiff_info_file, 'r') as json_file:
    #     data = json.load(json_file)
    #     upper_left_x = data["upper_left_x"]
    #     upper_left_y = data["upper_left_y"]
    #     lower_right_x = data["lower_right_x"]
    #     lower_right_y = data["lower_right_y"]
    #     width = data["width"]
    #     height = data["height"]
    #     xres = data["xres"]
    #     yres = data["yres"]

    # json object for storing the detection boxes
    # detectionJson = {
    #     "type": "FeatureCollection",
    #     "features" : []
    # }

    # for storing image info for overlap detection
    image_json = {}

    print("\nPerforming object detection:")
    prev_time = time.time()
    start_time = time.time()

    for batch_i, (img_paths, input_imgs) in enumerate(dataloader):
        # Configure input
        input_imgs = Variable(input_imgs.type(Tensor))

        # Get detections
        with torch.no_grad():
            detections = model(input_imgs)
            detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres)

        # Log progress
        current_time = time.time()
        inference_time = datetime.timedelta(seconds=current_time - prev_time)
        prev_time = current_time
        print("\t+ Batch %d, Inference Time: %s" % (batch_i, inference_time))

        # Save image and detections
        imgs.extend(img_paths)
        img_detections.extend(detections)

    total_time = time.time()

    # Bounding-box colors
    cmap = plt.get_cmap("tab20b")
    colors = [cmap(i) for i in np.linspace(0, 1, 20)]

    print("\nSaving images:")
    # Iterate through images and save plot of detections
    for img_i, (path, detections) in enumerate(zip(imgs, img_detections)):

        im = Image.open(path)
        image_width, image_height = im.size

        print("(%d) Image: '%s'" % (img_i, path))

        # image_name = path.replace("data/samples\\","")
        # detectionJson["features"][0][image_name] = {}

        # Create plot
        img = np.array(Image.open(path))

        # Draw bounding boxes and labels of detections
        if detections is not None:
            # Rescale boxes to original image
            detections = rescale_boxes(detections, opt.img_size, img.shape[:2])
            unique_labels = detections[:, -1].cpu().unique()
            n_cls_preds = len(unique_labels)
            bbox_colors = random.sample(colors, n_cls_preds)
            box_idx = 0
            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                if classes[int(cls_pred)] == "palm0":
                    print("\t+ Label: %s, Conf: %.5f" % (classes[int(cls_pred)], cls_conf.item()))

                    if x1 < 0:
                        x1 = torch.tensor(0)

                    if y1 < 0:
                        y1 = torch.tensor(0)

                    if x2 > im.width:
                        x2 = torch.tensor(im.width)

                    if y2 > im.height:
                        y2 = torch.tensor(im.height)

                    box_w = x2 - x1
                    box_h = y2 - y1

                    center_x = ((x1.item() + x2.item()) / 2)
                    center_y = ((y1.item() + y2.item()) / 2)

                    # xmin_tiff = upper_left_x + (x1 / xres)
                    # ymin_tiff = upper_left_y + (y1 / yres)
                    # xmax_tiff = upper_left_x + (x2 / xres)
                    # ymax_tiff = upper_left_y + (y2 / yres)
                    # box_w_tiff = abs(box_w / xres)
                    # box_h_tiff = abs(box_h / yres)

                    # hypotenuse = math.sqrt( (xmax_tiff-xmin_tiff) ** 2 + (ymax_tiff - ymin_tiff) ** 2 )
                    #
                    # upper_left_corner = [xmin_tiff.data.tolist(), ymin_tiff.data.tolist()]
                    # upper_right_corner = [xmax_tiff.data.tolist(), ymin_tiff.data.tolist()]
                    # lower_left_corner = [xmin_tiff.data.tolist(), ymax_tiff.data.tolist()]
                    # lower_right_corner = [xmax_tiff.data.tolist(), ymax_tiff.data.tolist()]

                    # boxname = "box" + str(box_idx)
                    # detectionJson["features"][0][image_name][boxname] = {
                    #     "
                    #     ": conf.data.tolist(),
                    #     "cls_conf": cls_conf.data.tolist(),
                    #     "cls_pred": cls_pred.data.tolist(),
                    #     "xmin": x1.data.tolist(),
                    #     "xmax": x2.data.tolist(),
                    #     "ymin": y1.data.tolist(),
                    #     "ymax": y2.data.tolist(),
                    #     "xmin_tiff": xmin_tiff.data.tolist(),
                    #     "ymin_tiff": ymin_tiff.data.tolist(),
                    #     "xmax_tiff": xmax_tiff.data.tolist(),
                    #     "ymax_tiff": ymax_tiff.data.tolist(),
                    #     "width": box_w.data.tolist(),
                    #     "height": box_h.data.tolist(),
                    #     "width_tiff": box_w_tiff.data.tolist(),
                    #     "height_tiff": box_h_tiff.data.tolist(),
                    #     "label": classes[int(cls_pred)]
                    # }

                    # detectionJson["features"].append(
                    # {
                    #     "type": "Feature",
                    #     "geometry": {
                    #         "type": "Polygon",
                    #         "coordinates": [
                    #             [
                    #                 upper_right_corner, upper_left_corner, lower_left_corner, lower_right_corner, upper_right_corner
                    #             ]
                    #         ],
                    #     },
                    #     "properties":
                    #     {
                    #         "conf": conf.data.tolist(),
                    #         "cls_conf": cls_conf.data.tolist(),
                    #         "cls_pred": cls_pred.data.tolist(),
                    #         "width_tiff": box_w_tiff.data.tolist(),
                    #         "height_tiff": box_h_tiff.data.tolist(),
                    #         "label": classes[int(cls_pred)],
                    #     }
                    # })

                    # print(classes[int(cls_pred)])
                    # sys.exit()

                    box_idx += 1
                    image_name = os.path.basename(path)

                    if image_name not in image_json:
                        image_json[image_name] = {}

                    tile_coords = get_tile_coordinates(image_name)
                    image_json[image_name]["tile"] = tile_coords

                    image_json[image_name]["box" + str(box_idx)] = {
                        "xmin": round(x1.item(), 3),
                        "ymin": round(y1.item(), 3),
                        "xmax": round(x2.item(), 3),
                        "ymax": round(y2.item(), 3),
                        "width": round(box_w.item(), 3),
                        "height": round(box_h.item(), 3),
                        "image_width": image_width,
                        "image_height": image_height,
                        "class": classes[int(cls_pred)],
                        "class_conf": round(cls_conf.data.tolist(), 3)
                    }

                    color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]

                    # Create a Rectangle patch
                    bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor="none")
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255,0,0), 2)
                    cv2.imwrite(f"output/samples/{os.path.basename(path)[:-4]}.png", img)

                    # Add label
                    plt.text(
                        x1,
                        y1,
                        s=classes[int(cls_pred)] + " " + str(round(cls_conf.data.tolist(), 2)),
                        color="white",
                        verticalalignment="top",
                        bbox={"color": color, "pad": 0},
                    )

    # with open("detections.json", 'w') as fp:
    #     json.dump(detectionJson, fp, indent=4)

    print("Total inference time: " + str(total_time - start_time))
    # print("Image json: " + str(image_json))

    with open("image_json.json", "w") as img_json:
        json.dump(image_json, img_json, indent=4)

    overlap_detect = overlap_detection.OverlapDetect([500, 500], image_json)
    start = time.time()
    overlap_detect.find_overlap()
    end = time.time()
    print("Time elapsed for overlap detection: " + str(end - start) + " seconds.")