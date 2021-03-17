import os
import cv2
import numpy as np
from os.path import join
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from utils.metrics import compute_miou
from utils.utils import dict_to_list
from itertools import compress


def visualize_background_model(gaussian, colorspace, scale, filters):
    """
    Creates a plot with the estimated background

    :param gaussian: gaussian model
    :param colorspace: colorspace of the image
    :param scale: resize factor of the original image
    :param filters: filters used in the bg removal
    """

    if gaussian.shape[2] == 2:
        plt.figure(figsize=(10, 3))

        plt.subplot(1, 2, 1)
        plt.imshow(gaussian[:, :, 0], 'gray')
        plt.title('Mean')

        plt.subplot(1, 2, 2)
        ax = plt.gca()
        im = ax.imshow(gaussian[:, :, 1])
        plt.title('Variance')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)

    elif gaussian.shape[2] == 4:

        plt.figure(figsize=(10, 6))

        plt.subplot(2, 2, 1)
        plt.imshow(gaussian[:, :, 0], 'gray')
        plt.title('Mean')

        plt.subplot(2, 2, 2)
        ax = plt.gca()
        im = ax.imshow(gaussian[:, :, 1])
        plt.title('Variance')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)

        plt.subplot(2, 2, 3)
        plt.imshow(gaussian[:, :, 2], 'gray')
        plt.title('Mean')

        plt.subplot(2, 2, 4)
        ax = plt.gca()
        im = ax.imshow(gaussian[:, :, 3])
        plt.title('Variance')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)

    filters_name = list(compress(['laplacian', 'median', 'bilateral', 'denoise'], filters))

    plt.savefig(join('outputs', colorspace + '_' + str(scale) + '_'.join(filters_name) + '.jpg'))


def draw_bboxes(img, bboxes, color):
    """
    Draw bounding boxes onto an image
    :param img: Input RGB image of the scene
    :param bboxes: list of coordinates of the bounding boxes to be drawn
    :param color: color used to draw the bounding boxes
    :return: image with the bounding boxes drawn
    """

    for bbox in bboxes:
        bbox = bbox.astype(int)
        img = cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
    return img


def visualize_background_iou(miou, std_iou, xaxis, frame, frame_id, bg, gt, dets, opt, axis=[536, 915]):
    """
    Creates a plot to visualize the IoU with its mean and std deviation.

    :param miou: mean IoU
    :param std_iou: standard deviation of the IoU
    :param xaxis: x axis of the plot
    :param frame: original image
    :param frame_id: id of the frame plotted
    :param bg: background/foreground estimated
    :param gt: ground truth
    :param dets: bbox detected
    :param opt: options/config used in the background removal process
    :param axis: list of the frame ids evaluated
    """

    pos = np.where(bg[:, :, 0])
    frame[pos + (np.zeros(pos[0].shape, dtype=np.uint64),)] = 255
    frame[pos + (np.ones(pos[0].shape, dtype=np.uint64),)] = 191
    frame[pos + (np.ones(pos[0].shape, dtype=np.uint64) * 2,)] = 0

    if frame_id in gt.keys():
        gt_frame = np.array(dict_to_list(gt[frame_id], False))

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = draw_bboxes(img, gt_frame * opt.resize_factor, (0, 255, 0))

        if frame_id in dets.keys():
            dets_frame = np.array(dict_to_list(dets[frame_id], False))
            mean, std = compute_miou(gt_frame, dets_frame, opt.resize_factor)

            img = draw_bboxes(img, dets_frame, (0, 0, 255))
        else:
            mean, std = 0, 0

        miou = np.hstack((miou, mean))
        std_iou = np.hstack((std_iou, std))
        plt.figure(figsize=(5, 6))

        plt.subplot(2, 1, 1)
        plt.imshow(img)
        plt.plot(0, 0, "-", c=(0, 1, 0), label='Ground Truth')
        plt.plot(0, 0, "-", c=(0, 0, 1), label='Detection')
        plt.legend(prop={'size': 8}, loc='lower right')

        xaxis = np.hstack((xaxis, int(frame_id)))

        plt.subplot(2, 1, 2)
        plt.plot(xaxis, miou, 'cadetblue', label='Mean IoU')
        plt.fill(np.append(xaxis, xaxis[::-1]), np.append(miou + std_iou, (miou - std_iou)[::-1]), 'powderblue',
                 label='STD IoU')
        plt.axis([axis[0], axis[1], 0, 1])
        plt.xlabel('Frame id', fontsize=10)
        plt.ylabel('IoU', fontsize=10)
        plt.legend(prop={'size': 8}, loc='lower right')

        filters_name = list(compress(['laplacian', 'median', 'bilateral', 'denoise'], [opt.laplacian,
                                                                                       opt.median_filter,
                                                                                       opt.bilateral_filter,
                                                                                       opt.pre_denoise]))

        save_path = join(opt.output_path, str(opt.task), str(opt.resize_factor),
                         str(opt.alpha) + '_' + '_'.join(filters_name))

        os.makedirs(save_path, exist_ok=True)

        plt.savefig(join(save_path, frame_id + '.png'))
        plt.close()

    return miou, std_iou, xaxis


def plot_map_alphas(map, alpha):
    plt.plot(alpha, map)
    plt.xlabel('Alpha')
    plt.ylabel('mAP')
    plt.title('Alpha vs mAP')
    plt.show()
