import sys
sys.path.insert(1, '../')

import numpy as np
import os
import cv2
import pickle
import time
from tqdm import tqdm
from models.sort import Sort
from utils.utils import return_bb, str_frame_id, update_data, pol2cart
from utils.metrics import compute_iou, interpolate_bb
from models.optical_flow import block_matching, MaskFlownetOF
import pyflow.pyflow as pyflow


def compute_tracking_overlapping(det_bboxes, frames_paths, alpha, ratio, minWidth, nOuterFPIterations, 
<<<<<<< HEAD
                                nInnerFPIterations, nSORIterations, colType, threshold = 0.5, 
                                interpolate = True, remove_noise = True, flow_method='pyflow'):
=======
                                nInnerFPIterations, nSORIterations, colType, threshold=0.5,
                                window_size=35, shift=5, stride=1, interpolate=False, remove_noise=True, 
                                flow_method='pyflow'):
>>>>>>> 1ebb1e64c8064d108858f8920399e80d0e93a6bd

    id_seq = {}
    #not assuming any order
    start_frame = int(min(det_bboxes.keys()))
    num_frames = int(max(det_bboxes.keys())) - start_frame + 1

    #init the tracking by  using the first frame 
    for value, detection in enumerate(det_bboxes[str_frame_id(start_frame)]):
        detection['obj_id'] = value
        id_seq.update({value: True})

    if flow_method == 'mask_flownet':
        flownet = MaskFlownetOF()

    #now, frame by frame, no assuming order nor continuity
    if flow_method == 'block_matching':
        path = os.path.join('../outputs/flow', flow_method, str(window_size))
        os.makedirs(path, exist_ok=True)
    else:
        path = os.path.join('../outputs/flow', flow_method)
        os.makedirs(path, exist_ok=True)

    for i in tqdm(range(start_frame, num_frames - 1),'Frames Overlapping Tracking'):
        img1 = cv2.imread(frames_paths[i-1])
        img2 = cv2.imread(frames_paths[i])

        if os.path.isfile(os.path.join(path, 'flow_' + str(i) +'.pkl')):
            with open(os.path.join(path, 'flow_' + str(i) +'.pkl'), 'rb') as f:
                flow = pickle.load(f)

            u = flow[:, :, 0]
            v = flow[:, :, 1]
        
        elif flow_method == 'pyflow':
            img1 = img1.astype(float) / 255.
            img2 = img2.astype(float) / 255.
            u, v, _ = pyflow.coarse2fine_flow(img1, img2, alpha[0], ratio[0], minWidth[0], 
                                                nOuterFPIterations[0], nInnerFPIterations[0], 
                                                nSORIterations[0], colType)
            flow = np.array([u, v])

            with open(os.path.join(path, 'flow_' + str(i) +'.pkl'), 'wb') as f:
                pickle.dump(flow.astype(np.float16), f)

        elif flow_method == 'mask_flownet':
            flow = flownet.get_optical_flow(img1, img2)

            with open(os.path.join(path, 'flow_' + str(i) +'.pkl'), 'wb') as f:
                pickle.dump(flow.astype(np.float16), f)

            u = flow[:, :, 0]
            v = flow[:, :, 1]


        elif flow_method == 'block_matching':
            flow = block_matching(img1, img2, window_size, shift, stride)

            with open(os.path.join(path, 'flow_' + str(i) +'.pkl'), 'wb') as f:
                    pickle.dump(flow.astype(np.float16), f)

            u = flow[:, :, 0]
            v = flow[:, :, 1]

        #init
        id_seq = {frame_id: False for frame_id in id_seq}
        
        for detection in det_bboxes[str_frame_id(i+1)]:
            active_frame = i 
            bbox_matched = False
<<<<<<< HEAD
            # get of for the bb
            OF_x = u[int(detection['bbox'][1]):int(detection['bbox'][3]),int(detection['bbox'][0]):int(detection['bbox'][2])]
            OF_y = v[int(detection['bbox'][1]):int(detection['bbox'][3]),int(detection['bbox'][0]):int(detection['bbox'][2])]
            mean_OF_x = np.mean(OF_x)
            mean_OF_y = np.mean(OF_y) 
            
            detection['bbox'] = [detection['bbox'][0]-mean_OF_x, detection['bbox'][1]-mean_OF_y, 
                        detection['bbox'][2]-mean_OF_x, detection['bbox'][3]-mean_OF_y] 

=======

            OF_x = u[int(detection['bbox'][1]):int(detection['bbox'][3]),int(detection['bbox'][0]):int(detection['bbox'][2])]
            OF_y = v[int(detection['bbox'][1]):int(detection['bbox'][3]),int(detection['bbox'][0]):int(detection['bbox'][2])]

            mag, ang = cv2.cartToPolar(OF_x.astype(np.float32), OF_y.astype(np.float32))
            #keep the values which is found the most for mag and ang
            uniques, counts = np.unique(mag, return_counts=True)
            mc_mag = uniques[counts.argmax()]
            uniques, counts = np.unique(ang, return_counts=True)
            mc_ang = uniques[counts.argmax()]
            x, y = pol2cart(mc_mag, mc_ang)

            detection['bbox_of'] = [detection['bbox'][0]-x, detection['bbox'][1]-y, 
                        detection['bbox'][2]-x, detection['bbox'][3]-y]
         
>>>>>>> 1ebb1e64c8064d108858f8920399e80d0e93a6bd
            #if there is no good match on previous frame, check n-1 up to n=5
            while (bbox_matched == False) and (active_frame >= start_frame) and ((i - active_frame)<5):
                candidates = [candidate['bbox'] for candidate in det_bboxes[str_frame_id(active_frame)]]
                #compare with all detections in previous frame
                #best match
                iou = compute_iou(np.array(candidates), np.array(detection['bbox_of']))
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
                                new_bb = interpolate_bb(return_bb(det_bboxes, (active_frame+j), matching_id), detection['bbox'],frames_skip-j+1)
                                update_data(det_bboxes, (active_frame+1+j),*new_bb,0,matching_id)
                        break
                    else: #try next best match
                        iou[np.argmax(iou)] = 0
                active_frame = active_frame - 1
            
            if not bbox_matched:
                #new object
                detection['obj_id'] = max(id_seq.keys()) + 1

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


def compute_tracking_kalman(det_bboxes): 
    '''
    Funtion to compute the tracking using Kalman filter
    :return: dictionary with the detections and the ids of each bbox computed by the tracking
    '''
    
    data_list = dict_to_list_track(det_bboxes)

    total_time = 0.0
    total_frames = 0
    out = []
    idx_frame = []
    colours = np.random.rand(32,3) #used only for display

    mot_tracker = Sort() #create instance of the SORT tracker

    det_bboxes_new = {}

    count = 0
    for idx, frame in tqdm(det_bboxes.items(),'Frames Kalman Tracking'): # all frames in the sequence
        
        colors = []

        dets = data_list[data_list[:,0]==count,1:6]
        #im = io.imread(join(data_path,idx)+'.png')

        start_time = time.time()
        trackers = mot_tracker.update(dets)
        cycle_time = time.time() - start_time
        total_time += cycle_time

        for track in trackers:
            det_bboxes_new = update_data(det_bboxes_new, idx, *track[:4], 1., track[4])

        count+=1

    return det_bboxes_new