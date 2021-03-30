import numpy as np
import cv2
from tqdm import tqdm
from models.sort import Sort
from utils.utils import return_bb, str_frame_id, update_data
from utils.metrics import compute_iou, interpolate_bb
from models.optical_flow import block_matching
import pyflow.pyflow as pyflow

def compute_tracking_overlapping(det_bboxes, frames_paths, alpha, ratio, minWidth, nOuterFPIterations, 
                                nInnerFPIterations, nSORIterations, colType, threshold = 0.5, 
                                interpolate = True, remove_noise = True):

    id_seq = {}
    #not assuming any order
    start_frame = int(min(det_bboxes.keys()))
    num_frames = int(max(det_bboxes.keys())) - start_frame + 1

    #init the tracking by  using the first frame 
    for value, detection in enumerate(det_bboxes[str_frame_id(start_frame)]):
        detection['obj_id'] = value
        id_seq.update({value: True})
    #now, frame by frame, no assuming order nor continuity
    for i in tqdm(range(start_frame, num_frames),'Frames Overlapping Tracking'):
        img1 = cv2.imread(frames_paths[i-1])
        img2 = cv2.imread(frames_paths[i])
        img1 = img1.astype(float) / 255.
        img2 = img2.astype(float) / 255.
        #pred_OF = block_matching(img1, img2, window_size, shift, stride)
        u, v, _ = pyflow.coarse2fine_flow(img1, img2, alpha[0], ratio[0], minWidth[0], 
                                        nOuterFPIterations[0], nInnerFPIterations[0], 
                                        nSORIterations[0], colType)

        #init
        id_seq = {frame_id: False for frame_id in id_seq}
        
        for detection in det_bboxes[str_frame_id(i+1)]:
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
                                new_bb = interpolate_bb(return_bb(det_bboxes, (active_frame+j), matching_id), detection['bbox'],frames_skip-j+1)
                                update_data(det_bboxes, (active_frame+1+j),*new_bb,0,matching_id)
                        break
                    else: #try next best match
                        iou[np.argmax(iou)] = 0
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