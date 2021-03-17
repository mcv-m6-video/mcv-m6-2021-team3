import numpy as np
import cv2
from scipy.ndimage.morphology import binary_fill_holes
from utils.utils import close_odd_kernel


def filter_noise(bg, resize_factor, noise_filter=['base', False], min_area=0.0003):
    """
    Filters noise of the estimated foreground/background image

    :param bg: estimated background/foreground
    :param resize_factor: resize factor applied to the original frame
    :param noise_filter: noise filter to apply and wheter to filter by area the noise
    :param min_area: min area to consider as good a certain "object" on the image

    :return: the background model and the labels of the connected components
    """
    if noise_filter[0] in 'morph_filter':
        # Opening
        bg = cv2.morphologyEx(bg, cv2.MORPH_OPEN, close_odd_kernel(resize_factor * 2))
        # Closing
        bg = cv2.morphologyEx(bg, cv2.MORPH_CLOSE, close_odd_kernel(resize_factor * 6))

    # Connected components
    num_lab, labels = cv2.connectedComponents(bg)

    if noise_filter[1]:
        # Filter by area
        rm_labels = [u for u in np.unique(labels) if np.sum(labels == u) < min_area * bg.size]
        for label in rm_labels:
            bg[np.where(labels == label)] = 0
            labels[np.where(labels == label)] = 0

    return bg, labels


def generate_bbox(label_mask):
    """
    Generates bbox given lasks

    :return: the four components of a bounding box
    """
    pos = np.where(label_mask == 255)

    if sum(pos).size < 1:
        return 0, 0, 0, 0

    y = np.min(pos[0])
    x = np.min(pos[1])
    h = np.max(pos[0]) - np.min(pos[0])
    w = np.max(pos[1]) - np.min(pos[1])

    return x, y, w, h


def fill_and_get_bbox(labels, resize_factor, fill=True):
    """
    Applies filling operation to close components and gets bboxs in the image

    :return: the estimated background and its bboxes
    """
    bg_idx = np.argmax([np.sum(labels == u) for u in np.unique(labels)])
    fg_idx = np.delete(np.unique(labels), bg_idx)

    bg = np.zeros(labels.shape, dtype=np.uint8)

    bboxes = []

    for label in fg_idx:
        aux = np.zeros(labels.shape, dtype=np.uint8)
        aux[labels == label] = 255

        # Closing
        aux = cv2.morphologyEx(aux, cv2.MORPH_CLOSE, close_odd_kernel(resize_factor * 9))

        # Opening -> rm shadows
        aux = cv2.morphologyEx(aux, cv2.MORPH_OPEN, close_odd_kernel(resize_factor * 9))

        x, y, w, h = generate_bbox(aux)

        # Filter by overlap
        if np.sum(aux > 0) / (w * h) >= .5:

            if fill:
                # Fill holes
                aux = binary_fill_holes(aux).astype(np.uint8) * 255

            bg = bg + aux
            # Add new bbox
            bboxes.append([x, y, w, h])

    return bg, bboxes


def get_single_objs(bg, resize_factor, noise_filter='base', fill=True, mask=None):
    """
    Gets single objects in the image using openCV's connected components approach.

    :param bg: estimated background/foreground
    :param resize_factor: applied resize_factor to the original image
    :param noise_filter: string that selects the noise filter to apply
    :param fill: bool that controls whether to use the fill approach
    :param: mask: mask to apply to the image
    :return: estimated background and bboxes
    """
    if noise_filter:
        # Filter noise
        bg, labels = filter_noise(bg, resize_factor, noise_filter)
    else:
        _, labels = cv2.connectedComponents(bg)

    # Generate bboxes and (optional) fill holes
    bg, bboxes = fill_and_get_bbox(labels, resize_factor, fill, mask)

    return bg, bboxes
