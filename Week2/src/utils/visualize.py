import os
import cv2
import numpy as np
from os.path import join
import flow_vis
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from utils.metrics import compute_miou
from utils.utils import dict_to_list
import imageio
import random

def visualize_background_variance(bg_var):
    return

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

def visualize_background_iou(miou, std_iou, xaxis, frame, frame_id, bg, gt, dets, opt, axis=[536, 536+1606]):#850
    pos = np.where(bg[:,:,0])
    frame[pos+(np.zeros(pos[0].shape, dtype=np.uint64),)]=255
    frame[pos+(np.ones(pos[0].shape, dtype=np.uint64),)]=191
    frame[pos+(np.ones(pos[0].shape, dtype=np.uint64)*2,)]=0

    if frame_id in gt.keys():
        gt_frame = np.array(dict_to_list(gt[frame_id], False))

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = draw_bboxes(img, gt_frame*opt['resize_factor'], (0, 255, 0))

        if frame_id in dets.keys():
            dets_frame = np.array(dict_to_list(dets[frame_id], False))
            mean, std = compute_miou(gt_frame, dets_frame, opt['resize_factor'])

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

        plt.savefig(join('outputs', 'task_'+str(opt['task']), frame_id + '.png'))
        plt.close()

    return miou, std_iou, xaxis

def plot_map_alphas(map,alpha):

    plt.plot(alpha,map)
    plt.xlabel('Alpha')
    plt.ylabel('mAP')
    plt.title('Alpha vs mAP')
    plt.show()
