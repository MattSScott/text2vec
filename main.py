import math
import os

from skimage.io import imread
import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage.morphology import skeletonize
from skimage.filters import threshold_otsu


# we expect a list of images with name = letter to generate a dataset from
def gen_dataset(dataset_location) -> []:
    dataset = os.fsencode(dataset_location)
    for file in os.listdir(dataset):
        filename = os.fsdecode(file)
        if filename.endswith(".png"):
            full_path = dataset_location + '/' + filename
            img = imread(full_path, as_gray=True)
            resized_img = resize(img, (50, 50))
            skeleton = get_skeleton(resized_img)
            vector_set = get_vector_set(skeleton)
            path_set = get_path_set(vector_set)
            plt.figure()
            plt.imshow(skeleton)
            for path in path_set:
                for pt in path:
                    plt.plot(pt[0], pt[1], 'ro')
            plt.show()
            continue
    return []


def get_skeleton(image):
    threshold = threshold_otsu(image)
    binary_image = image > threshold
    binary_image = 1 - binary_image
    skeleton = skeletonize(binary_image)
    return skeleton


def get_vector_set(skel):
    vec_set = []
    for y in range(len(skel)):
        for x in range(len(skel[0])):
            if skel[y][x]:
                vec_set.append([x, y])
    return vec_set


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

    if len(terminal_nodes) > 0:
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
        else:
            all_paths.append(curr_path)
            curr_path = [next_point]
        vector_set_copy.remove(next_point)
        curr_point = next_point
    all_paths.append(curr_path)
    return all_paths


def closest_point(curr_node, nodes):
    closest = nodes[0]
    closest_dist = math.dist(closest, curr_node)

    for node in nodes[1:]:
        curr_dist = math.dist(node, curr_node)
        if curr_dist < closest_dist:
            closest = node
            closest_dist = curr_dist

    return closest


if __name__ == '__main__':
    db_loc = "datasets/database1"
    gen_dataset(db_loc)
