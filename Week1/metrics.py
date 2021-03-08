import os
from os.path import join
import png
import numpy as np
from utils import dict_to_list, gen_noisy_bbox, update_data
import voc_eval

def vec_error(gt, det, nchannel=3):
    
    dist = det[:,:,:nchannel]-gt[:,:,:nchannel]
    error = np.sqrt(np.sum(dist**2,axis=2))

    # discard vectors which from occluded areas (occluded = 0)
    non_occluded_idx = gt[:, :, 2] != 0

    return error[non_occluded_idx], error

def compute_error(gt=None, det=None, error=None, nchannel=3, op='mse', th=3):
    if error is None:
        assert gt is None, 'img1 is None'
        assert det is None, 'img2 is None'
        error = vec_error(gt, det, nchannel)[0]
    
    if op == 'mse':
        return np.mean(error)
    elif op == 'pep':
        return np.sum(error>th)/len(error)

def compute_iou(bb_gt, bb):
    '''
    iou = compute_iou(bb_gt, bb)
    Compute IoU between bboxes from ground truth and a single bbox.
    bb_gt: Ground truth bboxes
        Array of (num, bbox), num:number of boxes, bbox:(xmin,ymin,xmax,ymax)
    bb: Detected bbox
        Array of (bbox,), bbox:(xmin,ymin,xmax,ymax)
    '''

    # intersection
    ixmin = np.maximum(bb_gt[:, 0], bb[0])
    iymin = np.maximum(bb_gt[:, 1], bb[1])
    ixmax = np.minimum(bb_gt[:, 2], bb[2])
    iymax = np.minimum(bb_gt[:, 3], bb[3])
    iw = np.maximum(ixmax - ixmin + 1., 0.)
    ih = np.maximum(iymax - iymin + 1., 0.)
    inters = iw * ih

    # union
    uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
            (bb_gt[:, 2] - bb_gt[:, 0] + 1.) *
            (bb_gt[:, 3] - bb_gt[:, 1] + 1.) - inters)

    return inters / uni

def compute_miou(gt_frame, dets_frame, frame_id):
    iou=[]
    for det in dets_frame:
        iou.append(np.max(compute_iou(gt_frame,det)))
    return np.sum(iou)/len(iou)

def compute_total_moiu(gt, dets, frames):
    miou = np.empty(0,)

    for frame in frames:
        if os.name == 'nt':
            frame = frame.replace(os.sep, '/')
        frame_id = (frame.split('/')[-1]).split('.')[0]

        if frame_id in gt.keys() and frame_id in dets.keys() and int(frame_id)>210:
            gt_frame = np.array(dict_to_list(gt[frame_id],False))
            dets_frame = np.array(dict_to_list(dets[frame_id],False))
            
            miou = np.hstack((miou,compute_miou(gt_frame,dets_frame,frame_id)))
    
    return (np.sum(miou)/len(miou))


def single_noise_eval(imagenames, gt, x_size = 1920, y_size = 1080, bbox_generate = False, 
                      bbox_delete = False, random_noise = False, bbox_displacement = False, 
                      max_random_px = 5, max_displacement_px = 5,  max_perc_create_bbox = 0.5, 
                      max_prob_delete_bbox = 0.5): 
    dets = {}
    for frame, info in gt.items():
        gen_bbox = gen_noisy_bbox(dict_to_list(info), x_size, y_size, bbox_generate, bbox_delete,
                                  random_noise, bbox_displacement, max_random_px, max_displacement_px, 
                                  max_perc_create_bbox, max_prob_delete_bbox)
        
        for bbox in gen_bbox:
            dets = update_data(dets,int(frame),bbox[0],bbox[1],bbox[0]+bbox[2],bbox[1]+bbox[3],1.)
    
    _, _, ap = voc_eval.voc_eval(gt,imagenames,dets)

    miou = compute_total_moiu(gt, dets, imagenames)

    return [ap,miou]

def gen_noise_eval(imagenames, gt):
    # Displacement Evaluation 
    gen_eval = {}
    #Displacement of -+n pixels (random up to i) for each bbox
    result=[]
    r = range(1, 50, 3)
    for i in r:
        result.append(single_noise_eval(imagenames, gt, bbox_displacement = True, max_displacement_px = i))
    gen_eval['Displacement'] = [[*r], result.copy()]

    # Noisy  of -+n pixel (random up to i) for each w&h/bbox
    result=[]
    r = range(1, 50, 3)
    for i in r:
        result.append(single_noise_eval(imagenames, gt, random_noise = True, bbox_displacement = True, 
                                        max_displacement_px = 5,  max_random_px = i))
    gen_eval['Noise'] = [[*r], result.copy()]

    # Dropping bbox
    result=[]
    r = range(1, 100, 3)
    for i in r:
        result.append(single_noise_eval(imagenames, gt, bbox_delete = True, max_prob_delete_bbox = i/100))
    gen_eval['Delete'] = [[*r], result.copy()]

    # Add new "random" bboxes
    result=[]
    r = range(1, 100, 3)
    for i in r:
        result.append(single_noise_eval(imagenames, gt, bbox_generate = True, max_perc_create_bbox = i/100))
    gen_eval['Generate'] = [[*r], result.copy()]
    
    return gen_eval