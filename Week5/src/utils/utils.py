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
import png
import subprocess
from numpngw import write_png

def read_kitti_OF(flow_file):
    """
    Read from KITTI .png file
    :param flow_file: name of the flow file
    :return: optical flow data in matrix
    """
    
    flow_object = png.Reader(filename=flow_file)
    flow_direct = flow_object.asDirect()
    flow_data = list(flow_direct[2])
    (w, h) = flow_direct[3]['size']
    print("Reading %d x %d flow file in .png format" % (h, w))
    flow = np.zeros((h, w, 3), dtype=np.float64)

    for i in range(len(flow_data)):
        flow[i, :, 0] = flow_data[i][0::3]
        flow[i, :, 1] = flow_data[i][1::3]
        flow[i, :, 2] = flow_data[i][2::3]

    invalid_idx = (flow[:, :, 2] == 0)
    flow[:, :, 0:2] = (flow[:, :, 0:2] - 2 ** 15) / 64.0
    flow[invalid_idx, 0] = 0
    flow[invalid_idx, 1] = 0

    return flow

def write_png_flow(flow, png_file):
    flow = flow[:,:,:2]
    flow16 = (64*flow + 2**15).astype(np.uint16)
    imgdata = np.concatenate((flow16, np.ones(flow16.shape[:2] + (1,), dtype=flow16.dtype)), axis=2)
    
    write_png(png_file, imgdata)

    if exists(png_file):
        print('PNG file ' + colored('\'' + png_file + '\'', 'blue') + ' written successfully!')

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
            cv2.imwrite(join(folder, str(counter).zfill(4) + '.jpg'), frame)
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

def str_frame_id(id):
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

def return_bb(det_bboxes, frame, bb_id):
    for bbox in det_bboxes[str_frame_id(frame)]:
        if bbox['obj_id'] == bb_id:
            return bbox['bbox']
    return None

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)

    return(x, y)

def update_data(annot, frame_id, xmin, ymin, xmax, ymax, conf, obj_id=0):
    """
    Updates the annotations dict with by adding the desired data to it
    :param annot: annotation dict
    :param frame_id: id of the framed added
    :param xmin: min position on the x axis of the bbox
    :param ymin: min position on the y axis of the bbox
    :param xmax: max position on the x axis of the bbox
    :param ymax: max position on the y axis of the bbox
    :param conf: confidence
    :return: the updated dictionary
    """

    frame_name = '%04d' % int(frame_id)
    obj_info = dict(
        name='car',
        obj_id=obj_id,
        bbox=list(map(float, [xmin, ymin, xmax, ymax])),
        confidence=float(conf)
    )

    if frame_name not in annot.keys():
        annot.update({frame_name: [obj_info]})
    else:
        annot[frame_name].append(obj_info)

    return annot
