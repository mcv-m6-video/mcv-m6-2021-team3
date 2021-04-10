import os
import cv2
import png
import glob
import numpy as np
from tqdm import tqdm
from os.path import join, exists
import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split, KFold

from models.yolov3 import UltralyricsYolo, to_yolov3
from utils.utils import write_json_file, read_json_file, update_data
#from models.tracking import compute_tracking_overlapping, compute_tracking_kalman
from utils.metrics import voc_eval, compute_iou, compute_centroid, compute_total_miou, interpolate_bb

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
        frame_id, _, xmin, ymin, width, height, conf, _, _, _ = list(map(float, (frame.split('\n')[0]).split(',')))
        update_data(annot, frame_id, xmin, ymin, xmin + width, ymin + height, conf)
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
    def __init__(self, data_path, seq, output_path, det_name, extension='jpg', det_params=None):
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
        

        # OUTPUT PARAMETERS
        self.output_path = output_path        

        # Load detections and load frame paths and filter by gt
        self.gt_bboxes = {}
        self.det_bboxes = {}
        self.frames_paths = {}
        for cam in os.listdir(join(data_path,seq)):
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
    
    def data_to_model(self):
        to_yolov3(self.data, self.gt_bboxes, self.split[0])
    
    def detect(self):

        model = UltralyricsYolo(self.det_params['weights'], args=self.det_params)

        for cam, paths in self.cam_paths.items():
            print(self.det_params['mode']+f' for sequence: {self.seq}')
            if len(self.det_bboxes[cam]) == len(paths):
                continue
            for file_name in tqdm(paths, 'Model predictions ({}, {})'.format(cam, self.det_params['model'])):
                pred = model.predict(file_name)
                frame_id = file_name[-8:-4]
                for (bbox), conf in pred:
                    self.det_bboxes[cam] = update_data(self.det_bboxes[cam], frame_id, *bbox, conf)
        
            if self.save_json:
                write_json_file(self.det_bboxes[cam],join(self.output_path,self.seq,cam,self.det_name))

        return self.get_mAP()
    
    def tracking(self):
        if self.options.tracking_mode in 'overlapping':
            self.det_bboxes = compute_tracking_overlapping(self.det_bboxes, self.frames_paths,
                                                            self.alpha, self.ratio, self.minWidth, 
                                                            self.nOuterFPIterations, self.nInnerFPIterations, 
                                                            self.nSORIterations, self.colType,
                                                            flow_method=self.options.OF_mode,
                                                            window_size=self.window_size,
                                                            stride=self.stride,
                                                            shift=self.shift)
        elif self.options.tracking_mode in 'kalman':
            self.det_bboxes = compute_tracking_kalman(self.det_bboxes)

    def get_mAP(self):
        """
        Estimats the mAP using the VOC evaluation

        :return: map of all estimated frames
        """
        mAP50, mAP70 = [], []
        for gt_bboxes, cam_paths, det_bboxes in zip(self.gt_bboxes.values(), self.frames_paths.values(), self.det_bboxes.values()):
            print(self.det_params['mode']+f' mAP for sequence: {self.seq}')
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
