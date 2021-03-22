import os
import cv2
import glob
import numpy as np
from os.path import join
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from utils.metrics import compute_miou, compute_centroid
from utils.utils import dict_to_list, read_json_file, frame_id
from itertools import compress


def visualize_trajectories(path_in, path_out, det_bboxes):
    """
    """
    # not assuming any order
    start_frame = int(min(det_bboxes.keys()))
    num_frames = int(max(det_bboxes.keys())) - start_frame + 1

    id_ocurrence = {}
    # Count ocurrences and compute centers 
    for i in range(start_frame, num_frames):
        for detection in det_bboxes[frame_id(i)]:
            objt_id = detection['obj_id']
            if objt_id in id_ocurrence:
                id_ocurrence[objt_id].append((i,compute_centroid(detection['bbox'])))
            else:
                id_ocurrence[objt_id] = [(i,compute_centroid(detection['bbox']))] 
    # plot
    num_colors = 1000
    colours = np.random.rand(num_colors,3) 
    for f_id, frame in tqdm(det_bboxes.items(),"saving tracking img"):
        colors = []
        detections = []
        id_list = []
        for detection in frame:
            bb_id = detection['obj_id']
            bbbox = detection['bbox']
            detections.append(bbbox)
            id_list.append(bb_id)

        img = draw_frame_track(path_in, f_id, detections, colours, id_list, id_ocurrence) 
        cv2.imshow('Tracking',img)
        cv2.waitKey(1)
        cv2.imwrite(join(path_out,'tracking',f_id)+'.png',img)
        

def draw_frame_track(path_in, frame, detections, colors, ids, id_ocurrence=[]):
    """
    :param path_in: path where the frames are saved
    :param frame: frame id
    :param detections: the different detections achieved in each frame
    :param colors: colors needed to print the diferent bboxes
    :param fill: a boolean to decide if the bbox is filled or not
    :return: return the image created by the frame and its bboxes
    """
    img = cv2.imread(join(path_in,frame)+'.png')
    for detection, bb_id in zip(detections, ids):
        color = colors[bb_id%1000,:]*255
        img = cv2.rectangle(img, (int(detection[0]),int(detection[1])), (int(detection[2]),int(detection[3])), tuple(color), 3)
        img = cv2.putText(img, str(bb_id), (int(detection[0]),int(detection[1])-10),cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        if id_ocurrence:
            for track_id, tracking  in id_ocurrence.items():
                c_start = 0
                color = colors[track_id%1000,:]*255
                for f_id, c_end in tracking:
                    if f_id < int(frame):
                        if c_start:
                            img = cv2.line(img, c_start, c_end, color, 2)
                        c_start = c_end
    return img


def visualize_tracking(path_in, path_out, det_bboxes):
    """
    :param path_in: path where the frames are saved
    :param path_out: path where the images should be saved
    :param det_bboxes: dictionary with the information of the detections
    :param gt_bboxes: dictionary with the information of the gt
    """
    num_colors = 1000
    colours = np.random.rand(num_colors,3) 
    for f_id, frame in tqdm(det_bboxes.items(),"saving tracking img"):
        colors = []
        detections = []
        id_list = []
        for detection in frame:
            bb_id = detection['obj_id']
            bbbox = detection['bbox']
            detections.append(bbbox)
            id_list.append(bb_id)
            ec=colours[bb_id%num_colors,:]
            colors.append(ec)

        img = draw_frame_track(path_in, f_id, detections, colors, id_list) 
        cv2.imshow('Tracking',img)
        cv2.waitKey(1)
        cv2.imwrite(join(path_out,'tracking',f_id)+'.png',img)


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

