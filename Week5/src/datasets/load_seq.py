import os
import cv2
import png
import glob
import random
import numpy as np
from tqdm import tqdm
from os.path import join, exists, dirname
import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split, KFold

from modes.ultralytics_yolo import UltralyricsYolo, to_yolov3
from modes.tracking import compute_tracking_overlapping, compute_tracking_kalman, compute_tracking_iou
from utils.visualize import visualize_trajectories
from utils.utils import write_json_file, read_json_file, update_data, dict_to_list_IDF1, match_trajectories
from utils.metrics import voc_eval, compute_iou, compute_centroid, compute_total_miou, interpolate_bb, IDF1, compute_IDmetrics

import motmetrics as mm

def load_text(text_dir, text_name):
    """
    Parses an annotations TXT file
    :param xml_dir: dir where the file is stored
    :param xml_name: name of the file to parse
    :return: a dictionary with the data parsed
    """
    with open(join(text_dir, text_name), 'r') as f:
        txt = f.readlines()

    annot = {}
    for frame in txt:
        frame_id, bb_id, xmin, ymin, width, height, conf, _, _, _ = list(map(float, (frame.split('\n')[0]).split(',')))
        update_data(annot, frame_id-1, xmin, ymin, xmin + width, ymin + height, conf, int(bb_id))
    return annot

def load_xml(xml_dir, xml_name, ignore_parked=True):
    """
    Parses an annotations XML file
    :param xml_dir: dir where the file is stored
    :param xml_name: name of the file to parse
    :return: a dictionary with the data parsed
    """
    tree = ET.parse(join(xml_dir, xml_name))
    root = tree.getroot()
    annot = {}

    for child in root:
        if child.tag in 'track':
            if child.attrib['label'] not in 'car':
                continue
            obj_id = int(child.attrib['id'])
            for bbox in child.getchildren():
                '''if bbox.getchildren()[0].text in 'true':
                    continue'''
                frame_id, xmin, ymin, xmax, ymax, _, _, _ = list(map(float, ([v for k, v in bbox.attrib.items()])))
                update_data(annot, int(frame_id) + 1, xmin, ymin, xmax, ymax, 1., obj_id)

    return annot

def load_annot(annot_dir, name, ignore_parked=True):
    """
    Loads annotations in XML format or TXT
    :param annot_dir: dir containing the annotations
    :param name: name of the file to load
    :return: the loaded annotations
    """
    if name.endswith('txt'):
        annot = load_text(annot_dir, name)
    elif name.endswith('xml'):
        annot = load_xml(annot_dir, name, ignore_parked)
    else:
        assert 'Not supported annotation format ' + name.split('.')[-1]

    return annot


class LoadSeq():
    def __init__(self, data_path, seq, output_path, tracking_mode, det_name, extension='jpg', det_params=None):
        """
        Init of the Load Sequence class

        :param data_path: path to data
        :param args: configuration for the current estimation
        """
        
        # INPUT PARAMETERS
        self.data_path = data_path
        self.seq = seq
        self.det_params = det_params
        self.det_name = self.det_params['mode']+'_'+det_name
        if self.det_params['mode'] == 'tracking':
            self.det_name = 'inference_'+det_name
        self.track_mode = tracking_mode

        # OUTPUT PARAMETERS
        self.output_path = output_path        

        # Load detections and load frame paths and filter by gt
        self.gt_bboxes = {}
        self.det_bboxes = {}
        self.frames_paths = {}

        self.accumulators = {}

        for cam in os.listdir(join(data_path,seq)):
            if '.' in cam:
                continue

            # Load gt
            self.gt_bboxes.update({cam:load_annot(join(data_path,seq,cam), 'gt/gt.txt')})

            # Check if detections already computed
            json_path = join(output_path,seq,cam)
            os.makedirs(json_path,exist_ok=True)
            json_path = join(json_path,self.det_name)
            if exists(json_path):
                self.det_bboxes.update({cam:read_json_file(json_path)})
            else:
                self.det_bboxes.update({cam:{}})

            # Save paths to frames
            cam_paths = glob.glob(join(data_path,seq,cam,'vdo/*.'+extension))
            cam_paths = [path for frame_id,_ in self.gt_bboxes[cam].items() for path in cam_paths if frame_id in path]
            cam_paths.sort()
            self.frames_paths.update({cam:cam_paths})

            # Creat accumulator 
            self.accumulators.update({cam:mm.MOTAccumulator()})

    def train_val_split(self, split=.25, mode='test'):
        """
        Apply split to specific propotion of the dataset.
        """
        self.data = {}
        if mode in 'train':
            # Define cams used to train and validate
            cams = self.frames_paths.keys()
            cams_val = random.sample(cams, int(len(cams)*split))
            cams_train = list(set(cams)-set(cams_val))

            self.data.update({'train':dict(filter(lambda cam: cam[0] in cams_train, self.frames_paths.items()))})
            self.data.update({'val':dict(filter(lambda cam: cam[0] in cams_val, self.frames_paths.items()))})
        
        else:
            # The whole sequence used to test
            self.data.update({'test':self.frames_paths})
    
    def data_to_model(self, split=.25, mode='test'):
        self.train_val_split(split, mode)
        return to_yolov3(self.data, self.gt_bboxes)
    
    def detect(self):

        model = UltralyricsYolo(self.det_params['weights'], args=self.det_params)
        print(self.det_params['mode']+f' for sequence: {self.seq}')

        for cam, paths in self.frames_paths.items():    
            if len(self.det_bboxes[cam]) > 0:
                continue
            for file_name in tqdm(paths, 'Model predictions ({}, {})'.format(cam, self.det_params['model'])):
                pred = model.predict(file_name)
                frame_id = file_name[-8:-4]
                for (bbox), conf in pred:
                    self.det_bboxes[cam] = update_data(self.det_bboxes[cam], frame_id, *bbox, conf)
        
            write_json_file(self.det_bboxes[cam],join(self.output_path,self.seq,cam,self.det_name))

        return self.get_mAP()
    
    def tracking(self):
        self.ID_metrics={}

        if self.track_mode in 'overlapping':
            for cam, det_bboxes in self.det_bboxes.items():
                self.det_bboxes[cam] = compute_tracking_overlapping(det_bboxes)
                None
        elif self.track_mode in 'kalman':
            for cam, det_bboxes in self.det_bboxes.items():
                self.det_bboxes[cam] = compute_tracking_kalman(det_bboxes, self.gt_bboxes[cam], self.accumulators[cam], self.frames_paths)

                self.ID_metrics.update({cam:compute_IDmetrics(self.accumulators[cam])})
                print(f'Camera: {cam}')
                print(self.ID_metrics[cam])
        elif self.track_mode in 'iou_track':
            for cam, det_bboxes in self.det_bboxes.items():
                compute_tracking_iou(det_bboxes)



                
    def get_mAP(self):
        """
        Estimats the mAP using the VOC evaluation

        :return: map of all estimated frames
        """
        mAP50, mAP70 = [], []
        print(self.det_params['mode']+f' mAP for sequence: {self.seq}')
        for gt_bboxes, cam_paths, det_bboxes in zip(self.gt_bboxes.values(), self.frames_paths.values(), self.det_bboxes.values()):            
            mAP50.append(voc_eval(gt_bboxes, cam_paths, det_bboxes)[2])
            mAP70.append(voc_eval(gt_bboxes, cam_paths, det_bboxes, use_07_metric=True)[2])
        return np.mean(mAP50), np.mean(mAP70)

    def get_mIoU(self):
        """
        Estimates the mIoU

        :return: iou of all estimated frames
        """
        return \
            compute_total_miou(self.gt_bboxes, self.det_bboxes, self.frames_paths)

    def visualize(self):
        for (cam,cam_paths), det_bboxes in zip(self.frames_paths.items(), self.det_bboxes.values()):
            path_in = dirname(cam_paths[0])
            if self.det_params['mode'] == 'tracking':
                visualize_trajectories(path_in, join(self.output_path,self.seq,cam), det_bboxes)