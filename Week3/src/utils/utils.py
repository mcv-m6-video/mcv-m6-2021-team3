from os import makedirs
from os.path import exists, join
import glob
import numpy as np
import pathlib
import tqdm
import random
import json
from termcolor import colored
import imageio
import cv2
import subprocess

def write_json_file(data_dict, json_file):
    """
    Write json file
    :param data_dict: dict containing data to store
    :param json_file: name of the json file
    """

    with(open(json_file, 'w')) as f:
        json.dump(data_dict, f)

    if exists(json_file):
        print('Json file ' + colored('\'' + json_file + '\'', 'blue') + ' written successfully!')

def read_json_file(json_file):
    """
    Read json file
    :param json_file: name of the json file
    :return: dictionary with the json data
    """

    with(open(json_file, 'r+')) as f:
        data = json.load(f)

    if data is not None:
        print('Json file ' + colored('\'' + json_file + '\'', 'blue') + ' loaded successfully!')

    return data

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

    while capture.isOpened():
        retrieved, frame = capture.read()

        if retrieved:
            cv2.imwrite(join(folder, str(counter).zfill(4) + '.png'), frame)
            counter += 1
            progress_bar.update(1)
        else:
            print("End of video")
            break

    capture.release()

def read_txt_to_struct(fname):
    """
    Read txt to structure
    :param fname: filename 
    :return: the information in the fname into a list
    """
    data = []
    with open(fname, 'r') as fid:
        lines = fid.readlines()
        for line in lines:
            line = list(map(float, line.strip().split(',')))
            data.append(line)
    data = np.array(data)
    # change point-size format to two-points format
    data[:, 4:6] += data[:, 2:4]
    return data

def dict_to_list(frame_info, tlwh=True):
    """
    Transform a dictionary into a list
    :param frame_info: dictionary with the information needed to create the list
    :param tlwh: Boolean that determine if the format is top-left-width-height
    :return: return the list created
    """

    if tlwh:
        return [[obj['bbox'][0],
                 obj['bbox'][1],
                 obj['bbox'][2] - obj['bbox'][0],
                 obj['bbox'][3] - obj['bbox'][1]] for obj in frame_info]
    else:
        return [[obj['bbox'][0], obj['bbox'][1], obj['bbox'][2], obj['bbox'][3]] for obj in frame_info]

def dict_to_list_IDF1(data):
    """
    Transform a dictionary into a list with the format needed in IDF1 function
    :param data: dictionary with the information needed to create the list
    :return: return the list created (frame_idx, obj_id, bbox_coord, confidence, 3D point)
    """
    idf1_list = []

    for frame_id, frame in data.items():
        for detect in frame:
            idf1_list.append([float(frame_id),float(detect['obj_id']),float(detect['bbox'][0]),float(detect['bbox'][1]),float(detect['bbox'][2]), float(detect['bbox'][3]),float(detect['confidence'])])
    return np.array(idf1_list)

def dict_to_list_track(frame_info):
    """
    Transform a dictionary into a list
    :param frame_info: dictionary with the information needed to create the list
    :return: return the np array created with the information of the detection (frame_idx, bbox coord, confidence)
    """
    boxes = []
    for idx, obj in enumerate(frame_info):
        for bbox in frame_info[obj]:
            box_info = [idx, bbox['bbox'][0], bbox['bbox'][1], bbox['bbox'][2], bbox['bbox'][3], bbox['confidence']]
            boxes.append(box_info)
    return np.array(boxes)

def frames_to_gif(save_dir, ext):
    img_paths = glob.glob(join(save_dir, '*.' + ext))
    img_paths.sort()
    gif_dir = save_dir + '.gif'
    with imageio.get_writer(gif_dir, mode='I') as writer:
        for img_path in img_paths:
            image = imageio.imread(img_path)
            writer.append_data(image)

    print('Gif saved at ' + gif_dir)

def get_weights(model, framework):
    makedirs('data/weights/',exist_ok=True)
    if model.endswith('.pt') or model.endswith('.pkl'):
        model_path = model
    else:
        if framework in 'detectron2':
            model_path = 'data/weights/'+model+'.pkl'
        elif framework in 'ultralytics':
            model_path = 'data/weights/'+model+'.pt'
        
    if not exists(model_path):
        subprocess.call(['sh','./data/scripts/get_'+model+'.sh'])
    return model_path

def frame_id(id):
    return ('%04d' % id)

def bbox_overlap(ex_box, gt_box):
    '''
    Funtion to compute the intersection between the bboxes
    :param ex_box, gt_box: bbox (predicted, gt)
    :return: IoU
    '''
    ex_box = ex_box.reshape(-1, 4)
    gt_box = gt_box.reshape(-1, 4)
    paded_gt = np.tile(gt_box, [ex_box.shape[0], 1])
    insec = intersection(ex_box, paded_gt)

    uni = areasum(ex_box, paded_gt) - insec
    return insec / uni

def intersection(a, b):
    '''
    Funtion to compute the intersection points
    :param a, b: bbox (predicted, gt)
    :return: intersection points
    '''
    x = np.maximum(a[:, 0], b[:, 0])
    y = np.maximum(a[:, 1], b[:, 1])
    w = np.minimum(a[:, 2], b[:, 2]) - x
    h = np.minimum(a[:, 3], b[:, 3]) - y
    return np.maximum(w, 0) * np.maximum(h, 0)

def areasum(a, b):
    '''
    Funtion to compute the intersection area
    :param a,b: bbox (prediction, gt)
    :return: intersection area
    '''
    return (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]) + \
        (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])