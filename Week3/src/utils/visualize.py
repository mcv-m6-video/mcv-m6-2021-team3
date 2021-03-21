import os
import cv2
import glob
import numpy as np
from os.path import join
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from utils.metrics import compute_miou
from utils.utils import dict_to_list, read_json_file
from itertools import compress

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


def visualize_background_iou(data, segmen, gt, dets, framework, model, output_path, mode='inference', axis=[536, 915]):
    """
    Creates a plot to visualize the IoU with its mean and std deviation.

    :param data: path to train and val images
    :param segmen: segmentation estimated
    :param gt: ground truth
    :param dets: bbox detected
    :param opt: options/config used in the background removal process
    :param axis: list of the frame ids evaluated
    """
    miou, std_iou, xaxis = np.empty(0, ), np.empty(0, ), np.empty(0, )
    frames_paths = data['train']+data['val']
    frames_paths.sort()
    for file_name in tqdm(frames_paths, 'Saving predictions ({}, {})'.format(model, framework)):
        
        frame_id = file_name[-8:-4]
        if frame_id in gt.keys() and axis[0]<int(frame_id)<axis[1]:
            frame = cv2.imread(file_name)
            gt_frame = np.array(dict_to_list(gt[frame_id], False))

            if segmen is not None:
                pos = np.where(segmen[:, :, 0])
                frame[pos + (np.zeros(pos[0].shape, dtype=np.uint64),)] = 255
                frame[pos + (np.ones(pos[0].shape, dtype=np.uint64),)] = 191
                frame[pos + (np.ones(pos[0].shape, dtype=np.uint64) * 2,)] = 0

            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = draw_bboxes(img, gt_frame, (0, 255, 0))

            if frame_id in dets.keys():
                dets_frame = np.array(dict_to_list(dets[frame_id], False))
                mean, std = compute_miou(gt_frame, dets_frame)

                img = draw_bboxes(img, dets_frame, (0, 0, 255))
            else:
                mean, std = 0, 0

            miou = np.hstack((miou, mean))
            std_iou = np.hstack((std_iou, std))
            plt.figure(figsize=(5, 6))

            plt.subplot(2, 1, 1)
            img = cv2.resize(img, (int(img.shape[1]*.5),int(img.shape[0]*.5)))
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

            save_path = join(output_path, mode, model, framework)

            os.makedirs(save_path, exist_ok=True)

            plt.savefig(join(save_path, frame_id + '.png'))
            plt.close()

