import os
import cv2
import numpy as np
from os.path import join
import flow_vis
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from metrics import compute_miou
from utils import dict_to_list
import imageio

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


def visualize_iou(gt, dets, frames, det_model, save_dir='./task2'):
    """
    Plot the graphic of the IOU metric
    :param gt: list with the bounding boxes of the GT file
    :param dets: list with the detected bounding boxes
    :param frames: frames extracted from the video
    :param det_model: model used (Mask RCNN, SSD512, YOLO3)
    :param save_dir: path to where gif will be saved
    """
    os.makedirs(join(save_dir, det_model), exist_ok=True)

    gif_dir = join(save_dir, det_model + '.gif')

    if os.path.exists(gif_dir):
        print('Gif saved at ' + gif_dir)
        return

    miou, std_iou = np.empty(0, ), np.empty(0, )

    with imageio.get_writer(gif_dir, mode='I') as writer:

        for frame in tqdm(frames[499:800], 'Evaluating detections from {} at each frame'.format(det_model)):
            if os.name == 'nt':
                frame = frame.replace(os.sep, '/')
            frame_id = (frame.split('/')[-1]).split('.')[0]

            if frame_id in gt.keys():
                gt_frame = np.array(dict_to_list(gt[frame_id], False))
                dets_frame = np.array(dict_to_list(dets[frame_id], False))

                mean, std = compute_miou(gt_frame, dets_frame)
                miou = np.hstack((miou, mean))
                std_iou = np.hstack((std_iou, std))

                plt.figure(figsize=(5, 6))

                plt.subplot(2, 1, 1)
                img = cv2.cvtColor(cv2.imread(frame), cv2.COLOR_BGR2RGB)
                img = draw_bboxes(img, gt_frame, (0, 255, 0))
                img = draw_bboxes(img, dets_frame, (0, 0, 255))
                plt.imshow(img)
                plt.plot(0, 0, "-", c=(0, 1, 0), label='Ground Truth')
                plt.plot(0, 0, "-", c=(0, 0, 1), label='Detection')
                plt.legend(prop={'size': 8}, loc='lower right')

                xaxis = np.arange(500, 500 + len(miou), 1)

                plt.subplot(2, 1, 2)
                plt.plot(xaxis, miou, 'cadetblue', label='Mean IoU')
                plt.fill(np.append(xaxis, xaxis[::-1]), np.append(miou + std_iou, (miou - std_iou)[::-1]), 'powderblue',
                         label='STD IoU')
                plt.axis([500, 800, 0, 1])
                plt.xlabel('Frame id', fontsize=10)
                plt.ylabel('IoU', fontsize=10)
                plt.legend(prop={'size': 8}, loc='lower right')

                plt.savefig(join(save_dir, det_model, frame_id + '.png'))
                plt.close()

                image = imageio.imread(join(save_dir, det_model, frame_id + '.png'))
                writer.append_data(image)

    print('Gif saved at ' + gif_dir)
