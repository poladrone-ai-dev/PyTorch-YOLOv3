import os
import sys
import glob
import json
import pprint
import time
import re
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator
import xml.etree.ElementTree as ET

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

    def find_nth(self, haystack, needle, n):
        start = haystack.find(needle)
        while start >= 0 and n > 1:
            start = haystack.find(needle, start + len(needle))
            n -= 1
        return start

    def calculate_offset(self, base_coords, image_coords):
        x_offset = image_coords[0] - base_coords[0]
        y_offset = image_coords[1] - base_coords[1]
        return x_offset, y_offset

    def merge_detections(self, image_path, output_path):
        os.chdir(image_path)
        image_types = ['.png', '.jpg', '.PNG', '.JPG', '.tif']
        images = []
        fullpath_images = []

        for image in glob.glob("*"):
            if os.path.splitext(image)[1] in image_types:
                images.append(image)
                fullpath_images.append(os.path.join(image_path, image))

        # reference image for doing offset math
        base_coords = self.get_tile_coordinates(images[0])

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

            x_offset, y_offset = self.calculate_offset(base_coords, image_coords)

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
            x_offset, y_offset = self.calculate_offset(base_coords, image_coords)

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

    def draw_bbox(self, image_json):
        output_path = os.path.join(r'D:\PyTorch-YOLOv3-master\output\samples', 'redrawn_bbox')

        for image in image_json:
            img = np.array(Image.open(os.path.join(self.IMG_PATH, image)))
            plt.figure()
            fig, ax = plt.subplots(1)
            ax.imshow(img)
            for box in image_json[image]:
                if "box" in box:
                    x1 = image_json[image][box]["xmin"]
                    y1 = image_json[image][box]["ymin"]
                    box_w = image_json[image][box]["width"]
                    box_h = image_json[image][box]["height"]

                    bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor="red", facecolor="none")
                    ax.add_patch(bbox)

            plt.axis("off")
            plt.gca().xaxis.set_major_locator(NullLocator())
            plt.gca().yaxis.set_major_locator(NullLocator())

            if os.path.isdir(output_path) == False:
                os.mkdir(output_path)

            plt.savefig(os.path.join(output_path, image), bbox_inches="tight", pad_inches=0.0)
            plt.close()

    def get_tile_coordinates(self, filename):
        underscore_count = len(re.findall('_', filename))
        first_underscore = self.find_nth(filename, '_', underscore_count - 1)
        second_underscore = self.find_nth(filename, '_', underscore_count)
        dot_index = self.find_nth(filename, '.', 1)
        x_coord = int(filename[first_underscore + 1: second_underscore])
        y_coord = int(filename[second_underscore + 1: dot_index])
        coordinates = [x_coord, y_coord]
        return coordinates

    def get_image_name_from_tile(self, image, coords):
        first_underscore = self.find_nth(image, '_', 1)
        filename = image[:first_underscore]
        extension = image[-4:]
        image_name = filename + "_" + '0' + str(coords[0]) + '_' + '0' + str(coords[1]) + extension
        return image_name

    def find_neighbor(self, image_json, image):
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
        neighbor_tiles = []

        underscore_count = len(re.findall('_', image))
        first_underscore = self.find_nth(image, '_', underscore_count - 1)

        filename = image[:first_underscore]
        extension = image[-4:]

        for neighbor in neighbors:
            neighbor_name = ''
            neighbor_name = filename + "_" + str(neighbor[0]) + '_' + str(neighbor[1]) + extension

            if neighbor_name in image_json:
                neighbor_tiles.append(neighbor_name)

        return neighbor_tiles

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

    def correct_bbox_top_bottom(self, image, neighbor, box, neighbor_box):
        image_bbox_length = abs(self.image_json[image][box]["xmin"] - self.image_json[neighbor][neighbor_box]["xmin"])
        neighbor_bbox_length = abs(self.image_json[image][box]["xmax"] - self.image_json[neighbor][neighbor_box]["xmax"])

        if image_bbox_length > neighbor_bbox_length:
            self.image_json[neighbor][neighbor_box]["xmin"] = self.image_json[image][box]["xmin"]
            self.image_json[neighbor][neighbor_box]["xmax"] = self.image_json[image][box]["xmax"]
            self.image_json[neighbor][neighbor_bos]["width"] = self.image_json[image][box]["width"]
        else:
            self.image_json[image][box]["xmin"] = self.image_json[neighbor][neighbor_box]["xmin"]
            self.image_json[image][box]["xmax"] = self.image_json[neighbor][neighbor_box]["xmax"]
            self.image_json[image][box]["width"] = self.image_json[neighbor][neighbor_box]["width"]

    def correct_bbox_left_right(self, image, neighbor, box, neighbor_box):
        image_bbox_length = abs(self.image_json[image][box]["ymin"] - self.image_json[neighbor][neighbor_box]["ymin"])
        neighbor_bbox_length = abs(self.image_json[image][box]["ymax"] - self.image_json[neighbor][neighbor_box]["ymax"])

        if image_bbox_length > neighbor_bbox_length:
            self.image_json[neighbor][neighbor_box]["ymin"] = self.image_json[image][box]["ymin"]
            self.image_json[neighbor][neighbor_box]["ymax"] = self.image_json[image][box]["ymax"]
            self.image_json[neighbor][neighbor_box]["height"] = self.image_json[image][box]["height"]
        else:
            self.image_json[image][box]["ymin"] = self.image_json[neighbor][neighbor_box]["ymin"]
            self.image_json[image][box]["ymax"] = self.image_json[neighbor][neighbor_box]["ymax"]
            self.image_json[image][box]["height"] = self.image_json[neighbor][neighbor_box]["height"]


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

    def find_overlap(self):
        # TODO: the statistics for the overlapping objects can be used to calculate the x and y thresholds
        left_right_x_thresh = 10
        left_right_y_thresh = 50
        up_down_x_thresh = 50
        up_down_y_thresh = 10

        for image in self.image_json:

            # find the image's neighbors by the tile coordinates
            neighbors = self.find_neighbor(self.image_json, image)
            image_coords = self.get_tile_coordinates(image)

            for neighbor in neighbors:
                neighbor_coords = self.get_tile_coordinates(neighbor)
                neighbor_type = self.get_neighbor_position(image_coords, neighbor_coords)

                if neighbor_type == "right":
                    self.right_overlap(self.image_json, image, neighbor, left_right_x_thresh, left_right_y_thresh)

                elif neighbor_type == "left":
                    self.left_overlap(self.image_json, image, neighbor, left_right_x_thresh, left_right_y_thresh)

                elif neighbor_type == "top":
                    self.top_overlap(self.image_json, image, neighbor, up_down_x_thresh, up_down_y_thresh)

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
        self.draw_bbox(self.image_json)
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
