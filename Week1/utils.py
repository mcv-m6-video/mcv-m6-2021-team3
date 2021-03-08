import cv2
import glob
from os import makedirs
from os.path import join
import numpy as np
import pathlib
import tqdm
import random
from datasets import *

def read_video_file(video_file):
    """
    Read video from file
    :param video_file: name of the video file
    """

    path = pathlib.Path(video_file)
    folder = join(str(path.parent), path.name.split('.')[0])
    makedirs(folder, exist_ok=True)

    capture = cv2.VideoCapture(video_file)
    n_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    
    counter = 0
    progress_bar = tqdm.tqdm(range(n_frames), total=n_frames)

    while(capture.isOpened()):
        retrieved, frame = capture.read()
        
        if retrieved:
            cv2.imwrite(join(folder, str(counter).zfill(4) + '.png'), frame)
            counter += 1
            progress_bar.update(1)
        else:
            print("End of video")
            break
            
    capture.release()

def dict_to_list(frame_info, tlwh=True):
    if tlwh:
        return [[obj['bbox'][0],
                obj['bbox'][1], 
                obj['bbox'][2]-obj['bbox'][0],
                obj['bbox'][3]-obj['bbox'][1]] for obj in frame_info]
    else:
        return [[obj['bbox'][0], obj['bbox'][1], obj['bbox'][2], obj['bbox'][3]] for obj in frame_info]

def gen_noisy_bbox(list_bbox, x_size = 1920, y_size = 1080, bbox_generate = False,
                   bbox_delete = False, random_noise = False, bbox_displacement = False,
                   max_random_px = 5,   max_displacement_px = 5,  max_perc_create_bbox = 0.5, 
                   max_prob_delete_bbox = 0.5): 
   
    # assumes each bbox is a list ordered as [xmin, ymin, width, height]
    
    noisy_list_bbox = list_bbox.copy()
    prob_delete_bbox = random.random() * max_prob_delete_bbox 
    perc_create_bbox = random.random() * max_perc_create_bbox  
    
    num_bbox = len(list_bbox)
    new_generate_box = int(num_bbox*perc_create_bbox)
    
    # from image size
    max_ratio_bbox = 0.2
    # for width and height
    min_size_bbox_px = 10
    
    if bbox_delete:
        new_list_bbox = []
        for bbox in noisy_list_bbox:
            # deletes the perc_create_bbox % of the bboxes
            if random.random() > prob_delete_bbox:
                new_list_bbox.append(bbox)
        noisy_list_bbox = new_list_bbox
               
    for bbox in noisy_list_bbox:
        if random_noise:
            #width
            bbox[2] = bbox[2] + random.randint(-max_random_px, max_random_px)
            #height
            bbox[3] = bbox[3] + random.randint(-max_random_px, max_random_px)
        
        if bbox_displacement:
            #xmin
            bbox[0] = bbox[0] + random.randint(-max_displacement_px, max_displacement_px)
            #ymin
            bbox[1] = bbox[1] + random.randint(-max_displacement_px, max_displacement_px)
            
    if bbox_generate:
        for i in range(new_generate_box):
            width = max(int(x_size * max_ratio_bbox * random.random()), min_size_bbox_px)
            height = max(int(x_size * max_ratio_bbox * random.random()), min_size_bbox_px)
            xmin = random.randint(0, x_size - width)
            ymin = random.randint(0, x_size - height) 
            
            noisy_list_bbox.append([xmin, ymin, width, height])

    return noisy_list_bbox