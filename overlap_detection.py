import os
import sys
import glob
import json
import pprint
import time
import re
import cv2
import numpy as np
from PIL import Image
import threading
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator
import xml.etree.ElementTree as ET

Image.MAX_IMAGE_PIXELS = 100000000000

class OverlapDetect():

    def __init__(self, tile_offset, image_json = None, image_path=None, xml_path=None):
        self.IMG_PATH = r'D:\PyTorch-YOLOv3-master\data\samples'
        self.XML_PATH = r'D:\image_processing\output\tree_annotation'

        if image_json == None:
            self.image_json = {}
        else:
            self.image_json = image_json

        self.overlap_count = 0
        self.corner_case_count = 0
        self.non_overlap_count = 0
        self.left_right_overlap_dict = {}
        self.up_down_overlap_dict = {}
        self.x_offset = tile_offset[0]
        self.y_offset = tile_offset[1]
        self.base_coords = []

    # find the nth occurrence of a character in a string
    # param {string} haystack: target string
    # param {char} needle: target character
    # param {int} n: the nth occurrence
    # return {int} start: the index of the nth occurrence of target character
    def find_nth(self, haystack, needle, n):
        start = haystack.find(needle)
        while start >= 0 and n > 1:
            start = haystack.find(needle, start + len(needle))
            n -= 1
        return start

    def get_reference_image(self, image_path):
        os.chdir(image_path)
        image_types = ['.png', '.jpg', '.PNG', '.JPG', '.tif']
        images = []
        fullpath_images = []

        for image in glob.glob("*"):
            if os.path.splitext(image)[1] in image_types:
                images.append(image)
                fullpath_images.append(os.path.join(image_path, image))

        # reference image for doing offset math
        self.base_coords = self.get_tile_coordinates(images[0])

    # find the tile offset of an image from the reference image (top left image)
    # param {list} base_coords: the tile coordinates for reference image
    # param {list} image_coords: the tile coordinates for target image
    # return {int, int}: the x offset and y offset for target image
    def calculate_offset(self, base_coords, image_coords):
        x_offset = image_coords[0] - base_coords[0]
        y_offset = image_coords[1] - base_coords[1]
        return x_offset, y_offset

    # exports detection results for each image into individual files with corresponding names
    # param {dict} output_json: the json object to be exported
    def export_detection_result(self, output_json):
        output_path = os.path.join(r"D:\PyTorch-YOLOv3-master\output", "detection_output")

        if not os.path.isdir(output_path):
            os.mkdir(output_path)

        for image in output_json:
            image_name = os.path.splitext(image)[0]
            json.dump(output_json[image], open(os.path.join(output_path, image_name + '.json'), 'w'), indent=4)

    # adds offset to bounding box coordinates in image json
    # returns {dict} output_json: a copy of image_json with offsets added
    def add_offset_to_bbox_coords(self):
        output_json = self.image_json.copy()
        for image in output_json:
            image_coords = self.get_tile_coordinates(os.path.basename(image))
            x_offset, y_offset = self.calculate_offset(self.base_coords, image_coords)

            for box in output_json[image]:
                if "box" in box:
                    output_json[image][box]["xmin"] = round(output_json[image][box]["xmin"] + x_offset, 3)
                    output_json[image][box]["ymin"] = round(output_json[image][box]["ymin"] + y_offset, 3)
                    output_json[image][box]["xmax"] = round(output_json[image][box]["xmax"] + x_offset, 3)
                    output_json[image][box]["ymax"] = round(output_json[image][box]["ymax"] + y_offset, 3)

        return output_json

    def sort_image_array(self, image_names):
        image_coords = []
        sorted_image_names = []
        for image_name in image_names:
            image_coord = self.get_tile_coordinates(os.path.basename(image_name))
            image_coords.append(image_coord)

        image_coords = sorted(image_coords, key=lambda x: x[1])
        image_coords = sorted(image_coords, key=lambda x: x[0])

        underscore_count = len(re.findall('_', image_names[0]))
        first_underscore = self.find_nth(image_names[0], '_', underscore_count - 1)
        image_basename = image_names[0][:first_underscore]
        image_ext = os.path.splitext(image_names[0])[1]

        for image_coord in image_coords:
            sorted_image_names.append(image_basename + "_" + str(image_coord[0]) + "_" + str(image_coord[1]) + image_ext)

        return sorted_image_names

    # merges output detection pictures into one larger picture, using the image names for position indexing
    # param {string} image_path: path that contains the input images
    # param {string} output_path: path to store the merged image
    def merge_detections(self, image_path, output_path):
        os.chdir(image_path)
        image_types = ['.png', '.jpg', '.PNG', '.JPG', '.tif']
        images = []
        fullpath_images = []

        for image in glob.glob("*"):
            if os.path.splitext(image)[1] in image_types:
                images.append(image)
                # fullpath_images.append(os.path.join(image_path, image))

        images = self.sort_image_array(images)

        for image in images:
            fullpath_images.append(os.path.join(image_path, image))

        # reference image for doing offset math
        self.base_coords = self.get_tile_coordinates(os.path.basename(images[0]))
        print("base coordinates: " + str(self.base_coords))

        image_name = os.path.splitext(image)[0].split('_')[0]
        imgs = [Image.open(i) for i in fullpath_images]

        min_img_width = min(i.width for i in imgs)
        min_img_height = min(i.height for i in imgs)

        total_width = imgs[0].width
        total_height = imgs[0].height

        for i, img in enumerate(imgs):
            image_coords = self.get_tile_coordinates(os.path.basename(img.filename))

            if img.width > min_img_width:
                imgs[i] = img.resize((min_img_width, int(img.height / img.width * min_img_width)), Image.ANTIALIAS)
            if img.height > min_img_height:
                imgs[i] = img.resize((min_img_height, int(img.height / img.width * min_img_height)), Image.ANTIALIAS)

            x_offset, y_offset = self.calculate_offset(self.base_coords, image_coords)

            print([x_offset, y_offset])

            if x_offset > 0 and y_offset == 0:
                total_width += imgs[i].width
            elif x_offset == 0 and y_offset > 0:
                total_height += imgs[i].height

        img_merge = Image.new(imgs[0].mode, (total_width, total_height))

        x_base = 0
        y_base = 0
        x_dimen = 1
        y_dimen = 1

        for i in range(len(imgs)):
            image_coords = self.get_tile_coordinates(os.path.basename(imgs[i].filename))
            x_offset, y_offset = self.calculate_offset(self.base_coords, image_coords)

            if x_offset > 0 and y_offset == 0:
                x_dimen += 1

            if x_offset == 0 and y_offset > 0:
                y_dimen += 1

        img_idx = 0
        for x in range(x_dimen):
            for y in range(y_dimen):
                try:
                    img_merge.paste(imgs[img_idx], (x_base + x * imgs[i].width, y_base + y * imgs[i].height))
                    img_idx += 1
                except Exception:
                    pass

        img_merge = img_merge.convert("RGB")
        img_merge.save(os.path.join(output_path, image_name + '_merged.jpg'))

    def draw_bbox(self):
        output_path = os.path.join(r'D:\PyTorch-YOLOv3-master\output\samples', 'redrawn_bbox')

        for image in self.image_json:
            img = np.array(Image.open(os.path.join(self.IMG_PATH, image)))
            plt.figure()
            fig, ax = plt.subplots(1)
            ax.imshow(img)
            for box in self.image_json[image]:
                if "box" in box:
                    x1 = self.image_json[image][box]["xmin"]
                    y1 = self.image_json[image][box]["ymin"]
                    box_w = self.image_json[image][box]["width"]
                    box_h = self.image_json[image][box]["height"]

                    bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor="red", facecolor="none")
                    ax.add_patch(bbox)

            plt.axis("off")
            plt.gca().xaxis.set_major_locator(NullLocator())
            plt.gca().yaxis.set_major_locator(NullLocator())

            if os.path.isdir(output_path) == False:
                os.mkdir(output_path)

            plt.savefig(os.path.join(output_path, image), bbox_inches="tight", pad_inches=0.0)
            plt.close()

    # draws corrected bounding boxes from detection json files
    # param {string} json_path: path to the detection json file
    # param {string} output_path: path to the output image
    # param {string} output_image: output image for the detection
    def draw_corrected_bbox(self, json_path, output_path, output_image):
        os.chdir(json_path)
        json_datas = []

        # get_json_data_start = time.time()
        for json_file in glob.glob("*.json"):
            with open(json_file, 'r') as fp:
                json_data = json.load(fp)
                json_datas.append(json_data)

        # get_json_data_end = time.time()
        img = np.array(Image.open(output_image))

        # draw_box_start = time.time()
        for image in json_datas:
            for box in image:
                if "box" in box:
                    x1 = image[box]["xmin"]
                    y1 = image[box]["ymin"]
                    x2 = image[box]["xmax"]
                    y2 = image[box]["ymax"]
                    cls = image[box]["class"]
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2

                    # center point
                    # draw_circle_start = time.time()
                    # cv2.circle(img, (int(center_x), int(center_y)), 5, (0, 0, 255), 5)
                    # draw_circle_end = time.time()

                    # cv2.putText(img, cls, (int(center_x), int(center_y)), cv2.FONT_HERSHEY_SIMPLEX, \
                    #             1.0, (0, 0, 0), lineType=cv2.LINE_AA)

                    # draw bounding box
                    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 255), 2)
                    cv2.putText(img, cls, (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, \
                                1.0, (0, 0, 0), lineType=cv2.LINE_AA)

                    # imwrite_start = time.time()
                    #x = threading.Thread(target=cv2.imwrite, args=(os.path.join(output_path, os.path.basename(output_image)), img))
                    cv2.imwrite(os.path.join(output_path, os.path.basename(output_image)), img)
                    #x.start()
                    # imwrite_end = time.time()

                    # print("Draw circle duration: " + str(draw_circle_end - draw_circle_start))
                    # print("imwrite duration: " + str(imwrite_end - imwrite_start))

        # draw_box_end = time.time()
        # print("Get json data duration: " + str(get_json_data_end - get_json_data_start) + "s.")
        # print("Draw boxes duration: " + str(draw_box_end - draw_box_start) + "s.")

    # get the tile coordinates of target image
    # param {string} filename: name of the target file
    # returns {list} coordinates: tile coordinates of the image
    def get_tile_coordinates(self, filename):
        underscore_count = len(re.findall('_', filename))
        first_underscore = self.find_nth(filename, '_', underscore_count - 1)
        second_underscore = self.find_nth(filename, '_', underscore_count)
        dot_index = self.find_nth(filename, '.', 1)
        x_coord = int(filename[first_underscore + 1: second_underscore])
        y_coord = int(filename[second_underscore + 1: dot_index])
        coordinates = [x_coord, y_coord]
        return coordinates

    # get image name with the specified tile coordinates
    # param {string} image: the name of the reference image
    # param {list} coords: the tile coordinates of the new image
    # returns {string} image_name: name of the image
    def get_image_name_from_tile(self, image, coords):
        first_underscore = self.find_nth(image, '_', 1)
        filename = image[:first_underscore]
        extension = image[-4:]
        image_name = filename + "_" + '0' + str(coords[0]) + '_' + '0' + str(coords[1]) + extension
        return image_name

    # finds a list of existing neighbor images for target image
    # param {dict} image_json: json file containing the informations of all images
    # param {string} image: the name of the target image
    # returns {list} neighbor_images: a list of existing neighbor images
    def find_neighbor_images(self, image_json, image):
        image_tile = image_json[image]['tile']
        x_coord = image_tile[0]
        y_coord = image_tile[1]

        # columns first
        top_left = [x_coord - self.x_offset, y_coord - self.y_offset]
        top_right = [x_coord + self.x_offset, y_coord - self.y_offset]
        left = [x_coord - self.x_offset, y_coord]
        right = [x_coord + self.x_offset, y_coord]
        top = [x_coord, y_coord - self.y_offset]
        bottom = [x_coord, y_coord + self.y_offset]
        bottom_left = [x_coord - self.x_offset, y_coord + self.y_offset]
        bottom_right = [x_coord + self.x_offset, y_coord + self.y_offset]

        neighbors = [top_left, top, top_right, left, right, bottom_left, bottom, bottom_right]
        neighbor_images = []

        underscore_count = len(re.findall('_', image))
        first_underscore = self.find_nth(image, '_', underscore_count - 1)

        filename = image[:first_underscore]
        extension = image[-4:]

        for neighbor in neighbors:
            neighbor_name = ''
            neighbor_name = filename + "_" + str(neighbor[0]) + '_' + str(neighbor[1]) + extension

            if neighbor_name in image_json:
                neighbor_images.append(neighbor_name)

        return neighbor_images

    # gets the neighbor tile's position relative to a reference tile
    # param {list} image: tile coordinates of the reference image
    # param {list} neighbor: tile coordinates of the reference tile
    def get_neighbor_position(self, tile, neighbor):

        # column first
        if [tile[0] - self.x_offset, tile[1] - self.y_offset] == neighbor:
            return 'topleft'

        if [tile[0], tile[1] - self.y_offset] == neighbor:
            return 'top'

        if [tile[0] + self.x_offset, tile[1] - self.y_offset] == neighbor:
            return 'topright'

        if [tile[0] - self.x_offset, tile[1]] == neighbor:
            return 'left'

        if [tile[0] + self.x_offset, tile[1]] == neighbor:
            return 'right'

        if [tile[0] - self.x_offset, tile[1] + self.y_offset] == neighbor:
            return 'bottomleft'

        if [tile[0], tile[1] + self.y_offset] == neighbor:
            return 'bottom'

        if [tile[0] + self.x_offset, tile[1] + self.y_offset] == neighbor:
            return 'bottomright'

        return 'None'

    # looks at two mismatched neighbor boxes and takes the one with the higher width as the width
    # param {string} image: name of the target image
    # param {string} neighbor: neighbor image
    # param {string} box: name of the bounding box in target image
    # param {string} neighbor_box: name of the bounding box in neighbor image
    def correct_bbox_top_bottom(self, image, neighbor, box, neighbor_box):
        image_bbox_length = abs(self.image_json[image][box]["xmin"] - self.image_json[neighbor][neighbor_box]["xmin"])
        neighbor_bbox_length = abs(self.image_json[image][box]["xmax"] - self.image_json[neighbor][neighbor_box]["xmax"])

        if image_bbox_length > neighbor_bbox_length:
            self.image_json[neighbor][neighbor_box]["xmin"] = self.image_json[image][box]["xmin"]
            self.image_json[neighbor][neighbor_box]["xmax"] = self.image_json[image][box]["xmax"]
            self.image_json[neighbor][neighbor_box]["width"] = self.image_json[image][box]["width"]

            # self.image_json[image][box]["xmax"] = self.image_json[neighbor][neighbor_box]["xmax"]
            # self.image_json[image][box]["ymax"] = self.image_json[neighbor][neighbor_box]["ymax"]
            # self.image_json[image][box]["height"] += self.image_json[neighbor][neighbor_box]["height"]
            # del self.image_json[neighbor]
        else:
            self.image_json[image][box]["xmin"] = self.image_json[neighbor][neighbor_box]["xmin"]
            self.image_json[image][box]["xmax"] = self.image_json[neighbor][neighbor_box]["xmax"]
            self.image_json[image][box]["width"] = self.image_json[neighbor][neighbor_box]["width"]

            # self.image_json[neighbor][neighbor_box]["xmax"] = self.image_json[image][box]["xmax"]
            # self.image_json[neighbor][neighbor_box]["ymax"] = self.image_json[image][box]["ymax"]
            # self.image_json[neighbor][neighbor_box]["height"] += self.image_json[image][box]["height"]
            # del self.image_json[neighbor]

    # looks at two mismatched neighbor boxes and takes the one with the higher height as the height
    # param {string} image: name of the target image
    # param {string} neighbor: neighbor image
    # param {string} box: name of the bounding box in target image
    # param {string} neighbor_box: name of the bounding box in neighbor image
    def correct_bbox_left_right(self, image, neighbor, box, neighbor_box):
        image_bbox_length = abs(self.image_json[image][box]["ymin"] - self.image_json[neighbor][neighbor_box]["ymin"])
        neighbor_bbox_length = abs(self.image_json[image][box]["ymax"] - self.image_json[neighbor][neighbor_box]["ymax"])

        if image_bbox_length > neighbor_bbox_length:
            self.image_json[neighbor][neighbor_box]["ymin"] = self.image_json[image][box]["ymin"]
            self.image_json[neighbor][neighbor_box]["ymax"] = self.image_json[image][box]["ymax"]
            self.image_json[neighbor][neighbor_box]["height"] = self.image_json[image][box]["height"]

            # self.image_json[image][box]["xmax"] = self.image_json[neighbor][neighbor_box]["xmax"]
            # self.image_json[image][box]["ymax"] = self.image_json[neighbor][neighbor_box]["ymax"]
            # self.image_json[image][box]["width"] += self.image_json[neighbor][neighbor_box]["width"]
            # del self.image_json[neighbor]
        else:
            self.image_json[image][box]["ymin"] = self.image_json[neighbor][neighbor_box]["ymin"]
            self.image_json[image][box]["ymax"] = self.image_json[neighbor][neighbor_box]["ymax"]
            self.image_json[image][box]["height"] = self.image_json[neighbor][neighbor_box]["height"]

            # self.image_json[neighbor][neighbor_box]["xmax"] = self.image_json[image][box]["xmax"]
            # self.image_json[neighbor][neighbor_box]["ymax"] = self.image_json[image][box]["ymax"]
            # self.image_json[neighbor][neighbor_box]["width"] += self.image_json[image][box]["width"]
            # del self.image_json[neighbor]

    def top_overlap(self, image_json, image, neighbor, up_down_x_thresh, up_down_y_thresh):
        for box in image_json[image]:
            if "box" in box:
                image_height = image_json[image][box]["image_height"]
                if image_json[image][box]["ymin"] <= up_down_y_thresh:
                    for neighbor_box in image_json[neighbor]:
                        if "box" in neighbor_box:
                            if abs(image_height - image_json[neighbor][neighbor_box]["ymax"]) <= up_down_y_thresh:
                                if abs(image_json[image][box]["xmin"] - image_json[neighbor][neighbor_box]["xmin"]) <= up_down_x_thresh and \
                                        abs(image_json[image][box]["xmax"] - image_json[neighbor][neighbor_box]["xmax"]) <= up_down_x_thresh and \
                                        image_json[image][box]["class"] == image_json[neighbor][neighbor_box]["class"]:

                                    self.correct_bbox_top_bottom(image, neighbor, box, neighbor_box)
                                    object = image_json[image][box]["class"]
                                    print("Found a top neighbor image.")
                                    print(box + " in " + image + " and " + neighbor_box + " in " + neighbor + " are the same. " +
                                                "The overlapping object is " + object + ".\n")
                                    if image not in self.up_down_overlap_dict:
                                        self.up_down_overlap_dict[image] = {
                                            "neighbor": [neighbor],
                                            "neighbor_type": ["top"],
                                            "box": [neighbor_box],
                                            "class": [object]
                                        }
                                        self.overlap_count += 1
                                    else:
                                        print(self.up_down_overlap_dict[image]["class"])
                                        self.up_down_overlap_dict[image]["neighbor"].append(neighbor)
                                        self.up_down_overlap_dict[image]["neighbor_type"].append("top")
                                        self.up_down_overlap_dict[image]["box"].append(neighbor_box)
                                        self.up_down_overlap_dict[image]["class"].append(object)
                                        self.overlap_count += 1

    def bottom_overlap(self, image_json, image, neighbor, up_down_x_thresh, up_down_y_thresh):
        for box in image_json[image]:
            if "box" in box:
                image_height = image_json[image][box]["image_height"]
                if abs(image_height - image_json[image][box]["ymax"]) <= up_down_y_thresh:
                    for neighbor_box in image_json[neighbor]:
                        if "box" in neighbor_box:
                            if (image_json[neighbor][neighbor_box]["ymin"]) <= up_down_y_thresh:
                                if abs(image_json[image][box]["xmin"] - image_json[neighbor][neighbor_box]["xmin"]) <= up_down_x_thresh and \
                                        abs(image_json[image][box]["xmax"] - image_json[neighbor][neighbor_box]["xmax"]) <= up_down_x_thresh and \
                                        image_json[image][box]["class"] == image_json[neighbor][neighbor_box]["class"]:

                                    self.correct_bbox_top_bottom(image, neighbor, box, neighbor_box)
                                    object = image_json[image][box]["class"]
                                    print("Found a bottom neighbor image.")
                                    print(box + " in " + image + " and " + neighbor_box + " in " + neighbor + " are the same. " +
                                                "The overlapping object is " + object + ".\n")
                                    if image not in self.up_down_overlap_dict:
                                        self.up_down_overlap_dict[image] = {
                                            "neighbor": [neighbor],
                                            "neighbor_type": ["bottom"],
                                            "box": [neighbor_box],
                                            "class": [object]
                                        }
                                        self.overlap_count += 1
                                    else:
                                        self.up_down_overlap_dict[image]["neighbor"].append(neighbor)
                                        self.up_down_overlap_dict[image]["neighbor_type"].append("bottom")
                                        self.up_down_overlap_dict[image]["box"].append(neighbor_box)
                                        self.up_down_overlap_dict[image]["class"].append(object)
                                        self.overlap_count += 1

    def left_overlap(self, image_json, image, neighbor, left_right_x_thresh, left_right_y_thresh):
        for box in image_json[image]:
            if "box" in box:
                image_width = image_json[image][box]["image_width"]
                if image_json[image][box]["xmin"] <= left_right_x_thresh:
                    for neighbor_box in image_json[neighbor]:
                        if "box" in neighbor_box:
                            if abs(image_width - image_json[neighbor][neighbor_box]["xmax"]) <= left_right_x_thresh:
                                if abs(image_json[image][box]["ymin"] - image_json[neighbor][neighbor_box]["ymin"]) <= left_right_y_thresh and \
                                        abs(image_json[image][box]["ymax"] - image_json[neighbor][neighbor_box]["ymax"]) <= left_right_y_thresh and \
                                        image_json[image][box]["class"] == image_json[neighbor][neighbor_box]["class"]:

                                    self.correct_bbox_left_right(image, neighbor, box, neighbor_box)
                                    object = image_json[image][box]["class"]
                                    print("Found a left neighbor image.")
                                    print(box + " in " + image + " and " + neighbor_box + " in " + neighbor + " are the same. " +
                                                "The overlapping object is " + object + ".\n")
                                    if image not in self.left_right_overlap_dict:
                                        self.left_right_overlap_dict[image] = {
                                            "neighbor": [neighbor],
                                            "neighbor_type": ["left"],
                                            "box": [neighbor_box],
                                            "class": [object]
                                        }
                                        self.overlap_count += 1
                                    else:
                                        self.left_right_overlap_dict[image]["neighbor"].append(neighbor)
                                        self.left_right_overlap_dict[image]["neighbor_type"].append("left")
                                        self.left_right_overlap_dict[image]["box"].append(neighbor_box)
                                        self.left_right_overlap_dict[image]["class"].append(object)
                                        self.overlap_count += 1

    def right_overlap(self, image_json, image, neighbor, left_right_x_thresh, left_right_y_thresh):
        for box in image_json[image]:
            if "box" in box:
                image_width = image_json[image][box]["image_width"]
                if abs(image_width - image_json[image][box]["xmax"]) <= left_right_x_thresh:
                    for neighbor_box in image_json[neighbor]:
                        if "box" in neighbor_box:
                            if image_json[neighbor][neighbor_box]["xmin"] <= left_right_x_thresh:
                                if abs(image_json[image][box]["ymin"] - image_json[neighbor][neighbor_box]["ymin"]) <= left_right_y_thresh and \
                                        abs(image_json[image][box]["ymax"] - image_json[neighbor][neighbor_box]["ymax"]) <= left_right_y_thresh and \
                                        image_json[image][box]["class"] == image_json[neighbor][neighbor_box]["class"]:

                                    self.correct_bbox_left_right(image, neighbor, box, neighbor_box)
                                    object = image_json[image][box]["class"]
                                    print("Found a right neighbor image.")
                                    print(box + " in " + image + " and " + neighbor_box + " in " + neighbor + " are the same. " +
                                                "The overlapping object is " + object + ".\n")
                                    if image not in self.left_right_overlap_dict:
                                        self.left_right_overlap_dict[image] = {
                                            "neighbor": [neighbor],
                                            "neighbor_type": ["right"],
                                            "box": [neighbor_box],
                                            "class": [object]
                                        }
                                        self.overlap_count += 1
                                    else:
                                        self.left_right_overlap_dict[image]["neighbor"].append(neighbor)
                                        self.left_right_overlap_dict[image]["neighbor_type"].append("right")
                                        self.left_right_overlap_dict[image]["box"].append(neighbor_box)
                                        self.left_right_overlap_dict[image]["class"].append(object)
                                        self.overlap_count += 1

    def bottomright_overlap(self, image_json, image, neighbor):
        for box in image_json[image]:
            if "box" in box:
                for neighbor_box in image_json[neighbor]:
                    if "box" in neighbor_box:
                        if image_json[image][box]["class"] == image_json[neighbor][neighbor_box]["class"]:
                            neighbor_xmin = image_json[neighbor][neighbor_box]["xmin"]
                            neighbor_ymin = image_json[neighbor][neighbor_box]["ymin"]

                            if neighbor_xmin <= 50 and neighbor_ymin <= 50:
                                print("Found a lower right neighbor.")
                                object = image_json[image][box]["class"]
                                print(box + " in " + image + " and " + neighbor_box + " in " + neighbor + " are the same. " +
                                            "The overlapping object is " + object + ".")

                                try:
                                    image_coords = self.get_tile_coordinates(image)
                                    right_neighbor_tile = [image_coords[0], image_coords[1] + self.y_offset]
                                    bottom_neighbor_tile = [image_coords[0] + self.x_offset, image_coords[1]]
                                    right_neighbor = self.get_image_name_from_tile(image, right_neighbor_tile)
                                    bottom_neighbor = self.get_image_name_from_tile(image, bottom_neighbor_tile)
                                    right_neighbor_index = self.left_right_overlap_dict[image]["neighbor"].index(right_neighbor)
                                    bottom_neighbor_index = self.up_down_overlap_dict[image]["neighbor"].index(bottom_neighbor)

                                    if self.left_right_overlap_dict[image]["neighbor_type"][right_neighbor_index] == "right" and \
                                            self.up_down_overlap_dict[image]["neighbor_type"][bottom_neighbor_index] == "bottom":
                                        self.corner_case_count += 1
                                except Exception:
                                    continue

    # find overlapping bounding boxes
    # returns {int}: returns the number of overlapping bounding boxes
    def find_overlap(self):

        self.get_reference_image(r'D:\PyTorch-YOLOv3-master\data\samples')

        # TODO: the statistics for the overlapping objects can be used to calculate the x and y thresholds
        left_right_x_thresh = 10
        left_right_y_thresh = 50
        up_down_x_thresh = 50
        up_down_y_thresh = 10

        for image in self.image_json:
            # find the image's neighbors by the tile coordinates
            neighbors = self.find_neighbor_images(self.image_json, image)
            image_coords = self.get_tile_coordinates(image)

            for neighbor in neighbors:
                neighbor_coords = self.get_tile_coordinates(neighbor)
                neighbor_type = self.get_neighbor_position(image_coords, neighbor_coords)

                if neighbor_type == "right":
                    self.right_overlap(self.image_json, image, neighbor, left_right_x_thresh, left_right_y_thresh)

                # elif neighbor_type == "left":
                #     self.left_overlap(self.image_json, image, neighbor, left_right_x_thresh, left_right_y_thresh)

                # elif neighbor_type == "top":
                #     self.top_overlap(self.image_json, image, neighbor, up_down_x_thresh, up_down_y_thresh)

                elif neighbor_type == "bottom":
                    self.bottom_overlap(self.image_json, image, neighbor, up_down_x_thresh, up_down_y_thresh)

                elif neighbor_type == "bottomright":
                    self.bottomright_overlap(self.image_json, image, neighbor)

            self.non_overlap_count += 1

        # print("Number of overlapping objects without considering corner cases: " + str(int(self.overlap_count / 2)))
        # print("Number of non-overlapping objects: " + str(self.non_overlap_count))
        print("Number of overlapping objects: " + str(int(self.overlap_count / 2) - 3 * self.corner_case_count))
        # print("Corner case count: " + str(self.corner_case_count))
        # print("Left right dict: " + str(self.left_right_overlap_dict) + '\n')
        # print("Up down dict: " + str(self.up_down_overlap_dict) + '\n')
        # self.draw_bbox()
        return int(self.overlap_count / 2) - 3 * self.corner_case_count

    def init_json(self):
        os.chdir(self.IMG_PATH)
        types = ('*.png', '*jpg', '*.PNG', '*.JPG')

        image_files = []
        for type in types:
            image_files.extend(glob.glob(type))

        for image in image_files:
            coordinates = self.get_tile_coordinates(image)
            self.image_json[image] = {
                "tile": coordinates,
            }

    def get_coordinates(self):
        os.chdir(self.XML_PATH)

        for f in glob.glob('*xml'):
            bbox_count = 0
            tree = ET.parse(f)
            filename = ""
            image_width = 0
            image_height = 0
            object_class = ""

            for child in tree.iter():

                if child.tag == "filename":
                    filename = child.text

                if child.tag == "width":
                    image_width = int(child.text)

                if child.tag == "height":
                    image_height = int(child.text)

                if child.tag == "object":
                    if child.find("name") != None:
                        object_class = child.find("name").text

                    if child.find("bndbox") != None:
                        bndbox = child.find("bndbox")
                        xmin = int(bndbox.find("xmin").text)
                        xmax = int(bndbox.find("xmax").text)
                        ymin = int(bndbox.find("ymin").text)
                        ymax = int(bndbox.find("ymax").text)
                        width = int(xmax - xmin)
                        height = int(ymax - ymin)

                        self.image_json[filename]["box" + str(bbox_count)] = {
                            "xmin": xmin,
                            "ymin": ymin,
                            "xmax": xmax,
                            "ymax": ymax,
                            "width": width,
                            "height": height,
                            "image_width": image_width,
                            "image_height": image_height,
                            "class": object_class
                        }
                        bbox_count += 1

if __name__ == "__main__":
    image_json = {}
    overlap_detect = OverlapDetect([1, 1])
    overlap_detect.init_json()
    overlap_detect.get_coordinates()
    pp = pprint.PrettyPrinter(depth=3)
    pp.pprint(image_json)
    start = time.time()
    overlap_detect.find_overlap()
    end = time.time()
    print("Time elapsed for overlap detection: " + str(end - start) + " seconds.")
