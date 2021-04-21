from os import makedirs
from os.path import exists, join
import glob
import numpy as np
import pathlib
import tqdm
import random
import json
import imageio
import cv2
import png
import yaml
import subprocess
from imageio import imread
from termcolor import colored
from numpngw import write_png
from scipy import ndimage

def read_kitti_OF(flow_file):
    """
    Read from KITTI .png file
    :param flow_file: name of the flow file
    :return: optical flow data in matrix
    """
    flow = cv2.imread(flow_file, cv2.IMREAD_UNCHANGED)
    flow = flow[:,:,::-1].astype(np.float64)
    flow = flow[:,:,:2]
    flow = (flow - 2**15) / 64.0

    return flow

def write_png_flow(flow, png_file):
    flow = flow[:,:,:2]
    flow16 = (64*flow + 2**15).astype(np.uint16)
    imgdata = np.concatenate((flow16, np.ones(flow16.shape[:2] + (1,), dtype=flow16.dtype)), axis=2)
    
    write_png(png_file, imgdata)

    if exists(png_file):
        print('PNG file ' + colored('\'' + png_file + '\'', 'blue') + ' written successfully!')


def write_yaml_file(yaml_dict, yaml_file):
    with open(yaml_file, 'w') as f:
        yaml.dump(yaml_dict, f)

    if exists(yaml_file):
        print('YAML file ' + colored('\'' + yaml_file + '\'', 'blue') + ' written successfully!')

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

def dict_to_array(data):
    """
    Transform a dictionary into a list with the format needed in IDF1 function
    :param data: dictionary with the information needed to create the list
    :return: return the list created (frame_idx, obj_id, bbox_coord, confidence, 3D point)
    """
    idf1_list = []

    for frame_id, frame in data.items():
        for detect in frame:
            if not detect['parked']:
                idf1_list.append([float(frame_id),float(detect['obj_id']),float(detect['bbox'][0]),float(detect['bbox'][1]),float(detect['bbox'][2]), float(detect['bbox'][3]),float(detect['confidence'])])
    return np.array(idf1_list)

def array_to_dict(array):
    """
    Transform a list into a dict with the format needed in the pipeline
    :param data: list with the information needed to create the dict
    :return: return the dict created
    """
    dets = {}
    for det in array:
        dets = update_data(dets,det[0], *det[2:], det[1])
    return dets
    

def dict_to_list_track(frame_info):
    """
    Transform a dictionary into a list
    :param frame_info: dictionary with the information needed to create the list
    :return: return the np array created with the information of the detection (frame_idx, bbox coord, confidence)
    """
    boxes = []
    for idx, obj in frame_info.items():
        for bbox in obj:
            if not bbox['parked']:
                box_info = [int(idx), bbox['bbox'][0], bbox['bbox'][1], bbox['bbox'][2], bbox['bbox'][3], bbox['confidence']]
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

def update_data(annot, frame_id, xmin, ymin, xmax, ymax, conf, obj_id=0, parked=False):
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
        confidence=float(conf),
        parked=parked
    )

    if frame_name not in annot.keys():
        annot.update({frame_name: [obj_info]})
    else:
        annot[frame_name].append(obj_info)

    return annot

def match_trajectories(det_bboxes, matches, in_out_ids):
    """
    Match ids between gt and predictions.
    """
    map_cam1 = {i:obj_id for i, obj_id in enumerate(in_out_ids[0])}
    map_cam2 = {i:obj_id for i, obj_id in enumerate(in_out_ids[1])}
    for matched in zip(*matches):
        det_bboxes[np.where(det_bboxes[:,1] == map_cam2[matched[1]]),1]=map_cam1[matched[0]]
    return det_bboxes

def compute_centroid(bb, resize_factor=1):
    """
    Computes centroid of bb
    :param bb: Detected bbox
    :return: Centroid [x,y] 
    """
    # intersection
    bb = np.array(bb) / resize_factor
    # (xmax - xmin)  / 2  
    x = (bb[2] + bb[0]) / 2
    # (ymax - ymin)  / 2  
    y = (bb[3] + bb[1]) / 2
    
    return (int(x), int(y))

def dist_to_roi(mask_path):
    roi = cv2.imread(mask_path,cv2.IMREAD_GRAYSCALE)/255
    return ndimage.distance_transform_edt(roi)

def filter_by_roi(det_bboxes, roi_dist, th=100):
    # Filter by proximity to roi area
    for frame_id, obj in det_bboxes.items():
        #new_obj = []
        for i, det in enumerate(obj):
            #xmin,ymin,xmax,ymax = det['bbox']
            #dist = np.min([roi_dist[int(y),int(x)] for x,y in [(xmin,ymin),(xmin,ymax),(xmax,ymin),(xmax,ymax)]])
            #if dist > th:
            centroid = compute_centroid(det['bbox'])
            if roi_dist[centroid[1],centroid[0]]>th:
                det_bboxes[frame_id][i].update({'parked':False})
            else:
                det_bboxes[frame_id][i].update({'parked':True})
                #det.update({'parked':False})
                #new_obj.append(det)
        #det_bboxes[frame_id] = new_obj.copy()

    return det_bboxes

def filter_static(det_bboxes):

    # remove cars which do not move much on its trajectory
    # compute frame history per id
    id_ocurrence = {}
    # Count ocurrences
    for idx_frame, detections in det_bboxes.items():
        for detection in detections:
            if not detection['parked']:
                obj_id = detection['obj_id']
                if obj_id in id_ocurrence:
                    id_ocurrence[obj_id].append(idx_frame)
                else:
                    id_ocurrence[obj_id] = [idx_frame]
    # Compute BB displacement from first to last frame
    id_distance = {}
    for det_id, frames in id_ocurrence.items():
        first_frame = frames[0]
        last_frame = frames[-1]
        first_bb = return_bb(det_bboxes,int(first_frame), int(det_id)) 
        last_bb = return_bb(det_bboxes,int(last_frame), int(det_id)) 
        distance = np.sum(np.array((np.array(compute_centroid(first_bb)) - np.array(compute_centroid(last_bb))))**2)
        id_distance.update({det_id: distance})
    # remove ids with distances below the threshold 
    ids_to_remove = [id_obj for id_obj in id_distance if id_distance[id_obj]<150]
    for idx_frame, detections in det_bboxes.items():
        for idx_bb, detection in enumerate(detections):
            if detection['obj_id'] in ids_to_remove and not detection['parked']:
                det_bboxes[idx_frame][idx_bb]['parked'] = True
    
    return det_bboxes

def color_hist(img_path, boxes, COLOR_SPACES, COLOR_RANGES, bins=25):

    img = cv2.imread(img_path)

    hist = np.empty([boxes.shape[0],len(COLOR_SPACES),bins,3])
    for i, (color_space, color_range) in enumerate(zip(COLOR_SPACES, COLOR_RANGES)):

        img_col = cv2.cvtColor(img, eval(color_space))
        
        for b, box in enumerate(boxes):
            img_box = img_col[box[1]:box[3],box[0]:box[2]]
            for c, c_r in enumerate(color_range):
                hist_c = cv2.calcHist([img_box],[c],None,[bins],c_r) #int((c_r[1]-c_r[0])*.1)
                hist[b,i,:,c] = cv2.normalize(hist_c, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX).squeeze()

    return hist

def read_txt_to_dict(txtpath):
    multitrack = {'c010':{},'c011':{},'c012':{},'c013':{},'c014':{},'c015':{}} 

    with open(txtpath, 'r') as fp:
        lines = fp.readlines()
        for line in lines:
            l = line.split(' ')
            cam = 'c0' + l[0]
            frame_id = int(l[2])
            xmin = int(l[3])
            ymin = int(l[4])
            xmax = xmin + int(l[5])
            ymax = ymin + int(l[6])
            conf = -1
            obj_id = int(l[1])
            multitrack[cam] = update_data(multitrack[cam],frame_id,xmin,ymin,xmax,ymax,conf,obj_id)

    return multitrack

