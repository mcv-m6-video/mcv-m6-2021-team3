import sys
sys.path.insert(1, '../')

import numpy as np
import os
import cv2
import pickle
import time
from tqdm import tqdm
from .sort import Sort
import matplotlib.pyplot as plt
from utils.utils import return_bb, str_frame_id, update_data, pol2cart, dict_to_list_track, write_png_flow
from utils.metrics import compute_iou, interpolate_bb, compute_dist_matrix, compute_iou
from .optical_flow import block_matching, MaskFlownetOF
from AIC2018.Tracking.ioutracker.iou_tracker import track_iou
#import pyflow.pyflow as pyflow

#otherwise it needs more than 4gbs to startup the model
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'

import matplotlib.pyplot as plt


def compute_tracking_overlapping(det_bboxes, frames_paths, threshold = 0.5, interpolate = True, remove_noise = True, flow_method = 'mask_flownet', save_img = True, cam = ''):

    id_seq = {}

    start_frame = int(min(det_bboxes.keys()))

    if flow_method == 'mask_flownet':
        flownet = MaskFlownetOF()
        if save_img:
            path = os.path.join('../outputs/flow', flow_method, cam)
            os.makedirs(path, exist_ok=True)

    #init the tracking by using the first frame which has at least one detection
    first_det_frame = start_frame
    while(len(id_seq)==0):
        obj_id = 0
        for detection in det_bboxes[str_frame_id(first_det_frame)]:
            if not detection['parked']:
                detection['obj_id'] = obj_id
                id_seq.update({obj_id: True})
                obj_id = obj_id +1
        first_det_frame = first_det_frame + 1

    #iterate frames
    for idx_frame, _ in tqdm(det_bboxes.items(),'Frames Overlapping Tracking'):
        id_seq = {obj_id: False for obj_id in id_seq}
        i = int(idx_frame)       
        if str_frame_id(int(idx_frame)+1) in det_bboxes.keys() and len(det_bboxes[str_frame_id(int(idx_frame)+1)])>0:
            #load frames for OF computation
            if flow_method == 'mask_flownet':
                img1 = cv2.imread(frames_paths[i])
                img2 = cv2.imread(frames_paths[i+1])  

                if os.path.isfile(os.path.join(path, 'flow_' + str(i) +'.png')):
                    flow = cv2.imread(os.path.join(path, 'flow_' + str(i) +'.png'))
               
                else:
                    flow = flownet.get_optical_flow(img1, img2)    
                    write_png_flow(flow, os.path.join(path, 'flow_' + str(i) +'.png'))

                u = flow[:,:,0]      
                v = flow[:,:,1]      

            for detection in det_bboxes[str_frame_id(int(idx_frame)+1)]:
                if not detection['parked']:
                    active_frame = i 
                    bbox_matched = False     
                    if flow_method == 'mask_flownet':
                        OF_x = u[int(detection['bbox'][1]):int(detection['bbox'][3]),int(detection['bbox'][0]):int(detection['bbox'][2])]
                        OF_y = v[int(detection['bbox'][1]):int(detection['bbox'][3]),int(detection['bbox'][0]):int(detection['bbox'][2])]

                        mag, ang = cv2.cartToPolar(OF_x.astype(np.float32), OF_y.astype(np.float32))
                        #keep the values which is found the most for mag and ang
                        uniques, counts = np.unique(mag, return_counts=True)
                        mc_mag = uniques[counts.argmax()]
                        uniques, counts = np.unique(ang, return_counts=True)
                        mc_ang = uniques[counts.argmax()]
                        x, y = pol2cart(mc_mag, mc_ang)

                        OF_bbox = [detection['bbox'][0]-x, detection['bbox'][1]-y, 
                                    detection['bbox'][2]-x, detection['bbox'][3]-y]
                    else: 
                        OF_bbox = detection['bbox'].copy()                
                    
                    #if there is no good match on previous frame, check n-1 up to n=5
                    while (bbox_matched == False) and (active_frame >= start_frame) and ((i - active_frame)<5):
                        candidates_bbox = [candidate['bbox'] for candidate in det_bboxes[str_frame_id(active_frame)] if candidate['parked']==False]
                        #compare with detections in previous frame
                        if len(candidates_bbox) > 0:
                            iou = compute_iou(np.array(candidates_bbox), np.array(OF_bbox))
                        else:
                            iou = 0
                        while np.max(iou) > threshold:
                            #candidate found, check if free
                            candidates_obj_id = [candidate['obj_id'] for candidate in det_bboxes[str_frame_id(active_frame)] if candidate['parked']==False]
                            matching_id = candidates_obj_id[np.argmax(iou)]
                            if id_seq[matching_id] == False:
                                detection['obj_id'] = matching_id
                                bbox_matched = True
                                #interpolate bboxes 
                                if i != active_frame and interpolate:
                                    frames_skip = i - active_frame
                                    for j in range(frames_skip):
                                        new_bb = interpolate_bb(return_bb(det_bboxes,(active_frame+j), matching_id), detection['bbox'], frames_skip-j+1)
                                        update_data(det_bboxes, (active_frame+1+j),*new_bb,0,matching_id, False)
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
        for idx_frame, detections in det_bboxes.items():
            if not detection['parked']:
                for detection in detections:
                    obj_id = detection['obj_id']
                    if obj_id in id_ocurrence:
                        id_ocurrence[obj_id] += 1
                    else:
                        id_ocurrence[obj_id] = 1
        # detections to be removed
        ids_to_remove = [id_obj for id_obj in id_ocurrence if id_ocurrence[id_obj]<4]
        for idx_frame, detections in det_bboxes.items():
            for idx_bb, detection in enumerate(detections):
                if detection['obj_id'] in ids_to_remove and not detection['parked']:
                    det_bboxes[idx_frame].pop(idx_bb)

    return det_bboxes

def compute_tracking_kalman(det_bboxes, gt_bboxes):#, accumulator): 
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

    for idx_frame, frame_det in tqdm(det_bboxes.items(), 'Frames Kalman Tracking'): # all frames in the sequence

        dets = data_list[data_list[:,0]==int(idx_frame),1:6]

        if dets.size==0:
            dets = []

        start_time = time.time()
        trackers = mot_tracker.update(dets)
        cycle_time = time.time() - start_time
        total_time += cycle_time

        if idx_frame in gt_bboxes.keys():
            if len(trackers)>0:
                for track in trackers:
                    det_bboxes_new = update_data(det_bboxes_new, idx_frame, *track[:4], 1., int(track[4]), False)
            for obj in frame_det:
                if obj['parked']:
                    det_bboxes_new = update_data(det_bboxes_new, idx_frame, *obj['bbox'], obj['confidence'], -1, True)
        
    return det_bboxes_new

def compute_tracking_iou(det_bboxes,cam, path):
    list_det_bboxes = []
    for detection in det_bboxes.values():
        list_det_bboxes.append(detection)

    tracking_dict = track_iou(list_det_bboxes, 0.2, 0.7, 0.5, 1, cam=cam, path=path)

    return tracking_dict
