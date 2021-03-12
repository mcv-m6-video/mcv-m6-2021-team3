import cv2
import numpy as np


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


def get_frame_background(frame, model, alpha=3, grayscale=True):
    """
    :param frame:
    :param model:
    :param grayscale:
    :return:
    """

    bg = np.zeros_like(frame)

    diff = frame - model[:, :, 0]
    foreground_idx = np.where(abs(diff) > alpha*(2 + model[:, :, 1]))

    bg[foreground_idx[0], foreground_idx[1]] = 255

    return bg


def read_frames(paths, grayscale=True):
    """
    :param paths:
    :param grayscale:
    :return:
    """

    images = []
    for file in paths:
        if grayscale:
            images.append(cv2.imread(file, cv2.IMREAD_GRAYSCALE))
        else:
            images.append(cv2.imread(file, cv2.IMREAD_COLOR))

    return np.asarray(images)