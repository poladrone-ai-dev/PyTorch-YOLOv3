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
        self.IMG_PATH = r'D:\image_processing\output\tree'
        self.XML_PATH = r'D:\image_processing\output\tree_annotation'

        if image_json == None:
            self.image_json = {}
        else:
            self.image_json = image_json

        self.overlap_count = 0
        self.corner_case_count = 0
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

    # draws the bounding box on a specific image
    def draw_bbox_neighbor(self, image_json):
        output_path = os.path.join(self.IMG_PATH, 'neighbor_detections')

        for image in image_json:
            img = np.array(Image.open(os.path.join(self.IMG_PATH, image)))
            plt.figure()
            fig, ax = plt.subplots(1)
            ax.imshow(img)

            if image in self.left_right_overlap_dict:

                left_right_neighbors = self.left_right_overlap_dict[image]["neighbor"]
                for left_right_neighbor in left_right_neighbors:
                    index = self.left_right_overlap_dict[image]["neighbor"].index(left_right_neighbor)
                    box = self.left_right_overlap_dict[image]["box"][index]
                    x1 = image_json[left_right_neighbor][box]["xmin"]
                    y1 = image_json[image][box]["ymin"]
                    box_w = image_json[image][box]["width"]
                    box_h = image_json[image][box]["height"]
                    bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor="red", facecolor="none")
                    ax.add_patch(bbox)

            if image in self.up_down_overlap_dict:

                up_down_neighbors = self.up_down_overlap_dict[image]["neighbor"]
                for up_down_neighbor in up_down_neighbors:
                    index = self.up_down_overlap_dict[image]["neighbor"].index(up_down_neighbor)
                    box = self.up_down_overlap_dict[image]["box"][index]

                    x1 = image_json[up_down_neighbor][box]["xmin"]
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

    def draw_bbox(self, image_json):
        output_path = os.path.join(self.IMG_PATH, 'neighbor_detections')

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

        # row first
        # top_left = [x_coord - self.x_offset, y_coord - self.y_offset]
        # top = [x_coord - self.x_offset, y_coord]
        # top_right = [x_coord - self.x_offset, y_coord + self.y_offset]
        # left = [x_coord, y_coord - self.y_offset]
        # right = [x_coord, y_coord + self.y_offset]
        # bottom_left = [x_coord + self.x_offset, y_coord - self.y_offset]
        # bottom = [x_coord + self.x_offset, y_coord]
        # bottom_right = [x_coord + self.x_offset, y_coord + self.y_offset]

        ########################################################################
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
            neighbor_name = filename + "_" + '0' + str(neighbor[0]) + '_' + '0' + str(neighbor[1]) + extension
            print("Neighbor name: " + neighbor_name)

            if neighbor_name in image_json:
                neighbor_tiles.append(neighbor_name)

        print("neighbor tiles: " + str(neighbor_tiles))
        return neighbor_tiles

    # gets the neighbor tile's position relative to a reference tile
    # param {list} image: tile coordinates of the reference image
    # param {list} neighbor: tile coordinates of the reference tile
    def get_neighbor_position(self, image, neighbor):

        # row first
        # if [image[0] - self.x_offset, image[1] - self.y_offset] == neighbor:
        #     return 'topleft'
        #
        # if [image[0] - self.x_offset, image[1]] == neighbor:
        #     return 'top'
        #
        # if [image[0] - self.x_offset, image[1] + self.y_offset] == neighbor:
        #     return 'topright'
        #
        # if [image[0], image[1] - self.y_offset] == neighbor:
        #     return 'left'
        #
        # if [image[0], image[1] + self.y_offset] == neighbor:
        #     return 'right'
        #
        # if [image[0] + self.x_offset, image[1] - self.y_offset] == neighbor:
        #     return 'bottomleft'
        #
        # if [image[0] + self.x_offset, image[1]] == neighbor:
        #     return 'bottom'
        #
        # if [image[0] + self.x_offset, image[1] + self.y_offset] == neighbor:
        #     return 'bottomright'
        #######################################################################

        # column first
        if [image[0] - self.x_offset, image[1] - self.y_offset] == neighbor:
            return 'topleft'

        if [image[0], image[1] - self.y_offset] == neighbor:
            return 'top'

        if [image[0] + self.x_offset, image[1] - self.y_offset] == neighbor:
            return 'topright'

        if [image[0] - self.x_offset, image[1]] == neighbor:
            return 'left'

        if [image[0] + self.x_offset, image[1]] == neighbor:
            return 'right'

        if [image[0] - self.x_offset, image[1] + self.y_offset] == neighbor:
            return 'bottomleft'

        if [image[0], image[1] + self.y_offset] == neighbor:
            return 'bottom'

        if [image[0] + self.x_offset, image[1] + self.y_offset] == neighbor:
            return 'bottomright'

        return 'None'

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
                                        self.up_down_overlap_dict[image]["neighbor"].append(neighbor)
                                        self.up_down_overlap_dict[image]["neighbor_type"].append("top")
                                        self.up_down_overlap_dict[image]["box"].append(neighbor_box)
                                        self.image_json[image]["class"].append(object)
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

        print("Image json length: " + str(len(self.image_json)))
        print("Imagae json : " + str(self.image_json) )
        for image in self.image_json:
            print(image)
            # find the image's neighbors by the tile coordinates
            neighbors = self.find_neighbor(self.image_json, image)
            image_coords = self.get_tile_coordinates(image)

            print("Image coords: " + str(image_coords))
            print("Neighbor count: " + str(len(neighbors)))

            for neighbor in neighbors:
                neighbor_coords = self.get_tile_coordinates(neighbor)
                neighbor_type = self.get_neighbor_position(image_coords, neighbor_coords)

                print("Neighbor coords: " + str(neighbor_coords))
                print("Neighbor type: " + str(neighbor_type))

                if neighbor_type == "right":
                    self.right_overlap(self.image_json, image, neighbor, left_right_x_thresh, left_right_y_thresh)

                if neighbor_type == "left":
                    self.left_overlap(self.image_json, image, neighbor, left_right_x_thresh, left_right_y_thresh)

                if neighbor_type == "top":
                    self.top_overlap(self.image_json, image, neighbor, up_down_x_thresh, up_down_y_thresh)

                if neighbor_type == "bottom":
                    self.bottom_overlap(self.image_json, image, neighbor, up_down_x_thresh, up_down_y_thresh)

                if neighbor_type == "bottomright":
                    self.bottomright_overlap(self.image_json, image, neighbor)

        print("Number of overlapping objects without considering corner cases: " + str(int(self.overlap_count / 2)))
        print("Number of overlapping objects: " + str(int(self.overlap_count / 2) - 3 * (self.corner_case_count)))
        print("Corner case count: " + str(self.corner_case_count))
        print("Left right dict: " + str(self.left_right_overlap_dict) + '\n')
        print("Up down dict: " + str(self.up_down_overlap_dict) + '\n')

        # self.draw_bbox_neighbor(self.image_json, left_right_overlap_dict, up_down_overlap_dict)
        # self.draw_bbox(self.image_json)

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
