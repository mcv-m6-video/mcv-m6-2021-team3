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


def get_frame_background(frame, model, grayscale=True):
    """
    :param frame:
    :param model:
    :param grayscale:
    :return:
    """

    bg = np.zeros_like(frame)

    diff = frame - model[:, :, 0]
    foreground_idx = np.where(abs(diff) > 2.5*model[:, :, 1])

    bg[foreground_idx[0], foreground_idx[1]] = 255
    kernel = np.ones((5, 5), np.uint8)
    bg = cv2.erode(bg, kernel, iterations=1)

    return bg


def read_frames(paths, grayscale=True):
    """
    :param path:
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