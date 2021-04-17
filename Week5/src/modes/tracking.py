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
#from AIC2018.ReID.Post_tracking import parse_tracks, filter_tracks, extract_images
#import pyflow.pyflow as pyflow

import matplotlib.pyplot as plt

#from AIC2018.Tracking.ioutracker.iou_tracker import track_iou

def compute_tracking_overlapping(det_bboxes, threshold = 0.5, interpolate = True, remove_noise = True, flow_method = 'mask_flownet'):

    id_seq = {}

    start_frame = int(min(det_bboxes.keys()))

    if flow_method == 'mask_flownet':
        flownet = MaskFlownetOF()
    
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
        if str_frame_id(int(idx_frame)+1) in det_bboxes.keys():
            for detection in det_bboxes[str_frame_id(int(idx_frame)+1)]:
                if not detection['parked']:
                    active_frame = i 
                    bbox_matched = False
                    #if there is no good match on previous frame, check n-1 up to n=5
                    while (bbox_matched == False) and (active_frame >= start_frame) and ((i - active_frame)<5):
                        candidates_bbox = [candidate['bbox'] for candidate in det_bboxes[str_frame_id(active_frame)] if candidate['parked']==False]
                        #compare with detections in previous frame
                        if len(candidates_bbox) > 0:
                            iou = compute_iou(np.array(candidates_bbox), np.array(detection['bbox']))
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

    tracking_dict = track_iou(list_det_bboxes, 0.2, 0.7, 0.5, 1, cam, path)

    return tracking_dict

def compute_multitracking(args):
    
    # Read tracks
    tracks = parse_tracks(args.tracking_csv)

    # Filter tracks
    tracks = filter_tracks(tracks, args.size_th, args.mask)
   
    # Extract images
    # tracks = extract_images(tracks, args.video, args.size_th, args.dist_th, args.mask, args.img_dir)
    
    if tracks is None: 
        sys.exit()
    
    # Save track obj
    os.system('mkdir -p %s' % args.output)
    file_name = args.video.split('/')[-1].split('.')[0]
    with open(os.path.join(args.output, '%s.pkl'%file_name), 'wb') as f:
        pickle.dump(tracks, f, protocol=pickle.HIGHEST_PROTOCOL)
    dets = []
    for t in tracks:
        dets.append(t.dump())
    dets = np.concatenate(dets, axis=0)
    np.savetxt(os.path.join(args.output, '%s.csv'%file_name), dets, delimiter=',', fmt='%f')