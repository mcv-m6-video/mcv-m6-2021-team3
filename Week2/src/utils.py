import cv2
import numpy as np
from tqdm.auto import tqdm

def model_background(frames, grayscale=True):
    """
    :param path: path of the image to load
    :return: np.array
    """
    n_frames, height, width = frames.shape
    gaussian = np.zeros((height, width, 2))
    gaussian[:, :, 0] = np.mean(frames, axis=0)
    gaussian[:, :, 1] = np.std(frames, axis=0)

    return gaussian

def get_frame_background(frame, model, alpha=3, grayscale=True, rm_noise=False, fill=False):
    """
    :param frame:
    :param model:
    :param grayscale:
    :param rm_noise:
    :param fill:
    :return:
    """

    bg = np.zeros_like(frame)

    diff = frame - model[:, :, 0]
    foreground_idx = np.where(abs(diff) > alpha*(2 + model[:, :, 1]))

    bg[foreground_idx[0], foreground_idx[1]] = 255

    if rm_noise:
        bg = filter_noise(bg)
    if fill:
        bg = fill_gaps(bg)

    return bg


def read_frames(paths, grayscale=True):
    """
    :param paths:
    :param grayscale:
    :return:
    """

    images = []
    for file_name in tqdm(paths, 'Reading frames'):
        if grayscale:
            image = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
            images.append(cv2.Laplacian(image,cv2.CV_64F))
        else:
            images.append(cv2.imread(file_name, cv2.IMREAD_COLOR))

    return np.asarray(images)

def filter_noise(bg):
    num_lab, labels = cv2.connectedComponents(bg)
    rm_labels = [u for u in np.unique(labels) if np.sum(labels==u)<10]
    for label in rm_labels:
        bg[np.where(bg==label)] = 0
    return bg

def fill_gaps(bg):
    return