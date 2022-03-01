import math
import os
import re

import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage.morphology import skeletonize
from skimage.filters import threshold_otsu


# we expect a list of images with name = letter to generate a dataset from
def gen_dataset(dataset_location, resolution):
    dataset = os.fsencode(dataset_location)
    for file in os.listdir(dataset):
        filename = os.fsdecode(file)
        if filename.endswith(".png"):
            letter = get_letter(filename)
            full_path = dataset_location + '/' + filename
            img = imread(full_path, as_gray=True)
            resized_img = resize(img, (resolution, resolution))
            skeleton = get_skeleton(resized_img)
            vector_set = get_vector_set(skeleton)
            path_set = get_path_set(vector_set)
            # plt.imshow(skeleton)
            # show_path(path_set)
            write_vec_file(letter, path_set, dataset_location)


def write_vec_file(letter, paths, location):
    db_num = re.search('\d+$', location).group(0)
    output_path = f"datasets/vectorised{db_num}/{letter}.npz"
    file = open(output_path, "w")
    np.savez(output_path, *paths)
    file.close()


def show_path(path_set):
    for path in path_set:  # illustrate path taken
        for i in range(len(path) - 1):
            pt1 = path[i]
            pt2 = path[i + 1]
            x = [pt1[0], pt2[0]]
            y = [pt1[1], pt2[1]]
            plt.plot(x, y, marker='')


def get_letter(file):
    f_copy = file
    return f_copy.replace(".png", "")


def get_skeleton(image):
    threshold = threshold_otsu(image)
    binary_image = image > threshold  # convert image to binary
    binary_image = 1 - binary_image  # invert grayscale image for processing
    skeleton = skeletonize(binary_image)
    return skeleton


# get_skeleton returns a boolean mask - filter these points to extract the x-y coordinates
def get_vector_set(skel):
    vec_set = []
    for y in range(len(skel)):
        for x in range(len(skel[0])):
            if skel[y][x]:
                vec_set.append([x, y])
    return vec_set


# find endpoint to start at (the point with 1 neighbour)
def get_terminal_node(vec_set):
    neighbouring_pixel_dist = 2
    terminal_nodes = []
    for vec in vec_set:
        num_neighbours = 0
        for other in vec_set:
            if vec == other:
                continue
            if math.dist(vec, other) < neighbouring_pixel_dist:
                num_neighbours += 1
            if num_neighbours > 1:
                break

        if num_neighbours == 1:
            terminal_nodes.append(vec)

    if len(terminal_nodes) > 0:  # if there are multiple terminal nodes, pick the leftmost
        return min_x(terminal_nodes)

    return vec_set[0]


def min_x(node_list):
    leftmost_x = node_list[0][0]
    leftmost_node = node_list[0]
    for el in node_list:
        if el != leftmost_node:
            curr_x = el[0]
            if curr_x < leftmost_x:
                leftmost_x = curr_x
                leftmost_node = el
    return leftmost_node


# take the list of vector points in the skeleton and turn it into a set of continuous paths
def get_path_set(vector_set):
    all_paths = []
    vector_set_copy = vector_set.copy()
    curr_point = get_terminal_node(vector_set)
    curr_path = [curr_point]
    vector_set_copy.remove(curr_point)
    while len(vector_set_copy) > 0:
        next_point = closest_point(curr_point, vector_set_copy)
        if math.dist(curr_point, next_point) < 2:
            curr_path.append(next_point)
        else:  # the next closest point is too far away - split into a second path
            joining_point = closest_point(next_point, curr_path)  # join using the closest point in the existing path
            all_paths.append(np.array(curr_path))
            curr_path = [joining_point, next_point]
        vector_set_copy.remove(next_point)
        curr_point = next_point
    all_paths.append(np.array(curr_path))
    return all_paths


def closest_point(curr_node, nodes):
    closest = nodes[0]  # curr_node is not in nodes - hence just start with index 0
    closest_dist = math.dist(closest, curr_node)

    for node in nodes[1:]:
        curr_dist = math.dist(node, curr_node)
        if curr_dist < closest_dist:
            closest = node
            closest_dist = curr_dist

    return closest


def load_dataset(db_loc):
    return list(np.load(db_loc).values())


def gen_transform(sx, sy, tx, ty):  # return a homography matrix
    return np.array([[sx, 0, 0], [0, sy, 0], [tx, ty, 1]])


def apply_transform(paths, tform):  # using a homography matrix, we change the entire co-ordinate set
    new_paths = []
    for path in paths:
        augmented = np.c_[path, np.ones(len(path))]
        new_paths.append(np.matmul(augmented, tform)[:, :2])
    return new_paths


def process_string(str_in, writing_style, pt_size, resolution):
    plt.figure()
    plt.gca().invert_yaxis()
    x_scale = pt_size / resolution
    y_scale = x_scale
    for i in range(len(str_in)):
        letter = str_in[i]
        db_loc = f"datasets/vectorised{writing_style}/{letter}.npz"
        letter_path = load_dataset(db_loc)
        x_shift = i * pt_size
        y_shift = 0
        tform = gen_transform(x_scale, y_scale, x_shift, y_shift)
        new_letter_path = apply_transform(letter_path, tform)
        show_path(new_letter_path)
    plt.show()


if __name__ == '__main__':
    rez = 200  # keep global for string processing
    database_loc = "datasets/database1"
    gen_dataset(database_loc, resolution=rez)  # set resolution higher for cleaner edges at cost of speed
    string_input = "mcm"
    process_string(string_input, writing_style=1, pt_size=5, resolution=rez)
