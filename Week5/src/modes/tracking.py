import sys
sys.path.insert(1, '../')

import numpy as np
import os
import cv2
import pickle
import time
from tqdm import tqdm
from .sort import Sort
from utils.utils import return_bb, str_frame_id, update_data, pol2cart, dict_to_list_track
from utils.metrics import compute_iou, interpolate_bb, compute_dist_matrix, compute_iou
from .optical_flow import block_matching, MaskFlownetOF
#import pyflow.pyflow as pyflow

import matplotlib.pyplot as plt

def compute_tracking_overlapping(det_bboxes, threshold = 0.5, interpolate = False, remove_noise = False):

    id_seq = {}
    start_frame = int(min(det_bboxes.keys()))
    num_frames = int(max(det_bboxes.keys())) - start_frame + 1

    #init the tracking by  using the first frame 
    for value, detection in enumerate(det_bboxes[str_frame_id(start_frame)]):
        detection['obj_id'] = value
        id_seq.update({value: True})
    #now, frame by frame, no assuming order nor continuity
    for idx_frame, f_detections in tqdm(det_bboxes.items(),'Frames Overlapping Tracking'):
        id_seq = {frame_id: False for frame_id in id_seq}
        i = int(idx_frame)
        for detection in f_detections:
            active_frame = i 
            bbox_matched = False
            #if there is no good match on previous frame, check n-1 up to n=5
            while (bbox_matched == False) and (active_frame >= start_frame) and ((i - active_frame)<5):
                candidates = [candidate['bbox'] for candidate in det_bboxes[str_frame_id(active_frame)]]
                #compare with all detections in previous frame
                #best match
                iou = compute_iou(np.array(candidates), np.array(detection['bbox']))
                while np.max(iou) > threshold:
                    #candidate found, check if free
                    matching_id = det_bboxes[str_frame_id(active_frame)][np.argmax(iou)]['obj_id']
                    if id_seq[matching_id] == False:
                        detection['obj_id'] = matching_id
                        bbox_matched = True
                        #interpolate bboxes 
                        if i != active_frame and interpolate:
                            frames_skip = i - active_frame
                            for j in range(frames_skip):
                                new_bb = interpolate_bb(return_bb((active_frame+j), matching_id), detection['bbox'],frames_skip-j+1)
                                update_data(det_bboxes, (active_frame+1+j),*new_bb,0,matching_id)
                        break
                    else: #try next best match
                        iou[np.argmax(iou)] = 0
                
                active_frame = active_frame - 1
                #check if the given frame exist
                while(str_frame_id(active_frame) not in det_bboxes.keys() and active_frame>= start_frame):
                    active_frame = active_frame - 1
           
            if not bbox_matched:
                #new object
                detection['obj_id'] = max(id_seq.keys())+1

            id_seq.update({detection['obj_id']: True})
            
    # filter by number of ocurrences
    if remove_noise:
        id_ocurrence = {}
        # Count ocurrences
        for i in range(start_frame, num_frames):
            for detection in det_bboxes[str_frame_id(i)]:
                objt_id = detection['obj_id']
                if objt_id in id_ocurrence:
                    id_ocurrence[objt_id] += 1
                else:
                    id_ocurrence[objt_id] = 1
        # detectiosn to be removed
        ids_to_remove = [id_obj for id_obj in id_ocurrence if id_ocurrence[id_obj]<4]
        for i in range(start_frame, num_frames):
            for idx, detection in enumerate(det_bboxes[str_frame_id(i)]):
                if detection['obj_id'] in ids_to_remove:
                    det_bboxes[str_frame_id(i)].pop(idx)
    return det_bboxes


def compute_tracking_kalman(det_bboxes, gt_bboxes, accumulator, frames_paths): 
    '''
    Funtion to compute the tracking using Kalman filter
    :return: dictionary with the detections and the ids of each bbox computed by the tracking
    '''

    data_list = dict_to_list_track(det_bboxes)

    total_time = 0.0
    total_frames = 0
    out = []
    idx_frame = []

    mot_tracker = Sort() #create instance of the SORT tracker

    det_bboxes_new = {}

    for (idx_frame, frame_gt), frame_path in tqdm(zip(gt_bboxes.items(), frames_paths), 'Frames Kalman Tracking'): # all frames in the sequence

        dets = data_list[data_list[:,0]==int(idx_frame),1:6]
        #im = io.imread(join(data_path,idx)+'.png')

        if dets.size==0:
            trackers=[]
        else:
            start_time = time.time()
            trackers = mot_tracker.update(dets)
            cycle_time = time.time() - start_time
            total_time += cycle_time

        if len(trackers)>0:
            for track in trackers:
                det_bboxes_new = update_data(det_bboxes_new, idx_frame, *track[:4], 1., int(track[4]))
            dists = compute_dist_matrix(det_bboxes_new[idx_frame], frame_gt)
            det_ids = [det['obj_id'] for det in det_bboxes_new[idx_frame]]
        else:
            dists=[]
            det_ids=[]

        gt_ids = [gt['obj_id'] for gt in frame_gt]
            
        accumulator.update(gt_ids, det_ids, dists, frameid=int(idx_frame), vf='')

    return det_bboxes_new