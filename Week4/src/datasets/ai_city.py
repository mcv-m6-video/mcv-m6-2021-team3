from os.path import join, basename, exists
import glob
from tqdm import tqdm
from sklearn.model_selection import train_test_split, KFold
import xml.etree.ElementTree as ET
import random

import matplotlib.pyplot as plt
import matplotlib.patches as patches
#matplotlib inline
from IPython import display as dp

import numpy as np
import matplotlib.pyplot as plt
import json
import cv2

import numpy as np
from skimage import io
import os
import time

from utils.metrics import voc_eval, compute_iou, compute_centroid, compute_total_miou, interpolate_bb
from utils.utils import write_json_file, read_json_file, str_frame_id, dict_to_list_IDF1, dict_to_list_track, update_data
from utils.visualize import visualize_background_iou

#from models.yolov3 import UltralyricsYolo, to_yolov3
from models.tracking import compute_tracking_overlapping, compute_tracking_kalman


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

class AICity:
    """
    This class contains all the logic of the background estimation process for the AICity dataset
    """

    def __init__(self, args):
        """
        Init of the AICity class

        :param args: configuration for the current estimation
        """
        self.options = args

        # INPUT PARAMETERS
        self.data_path = args.data_path
        self.img_size = args.img_size
        self.split = args.split
        self.model = args.model
        self.framework = args.framework
        self.mode = args.mode

        # Load detections
        self.gt_bboxes = load_annot(args.gt_path, 'ai_challenge_s03_c010-full_annotation.xml')
        self.det_bboxes = read_json_file('yolov3_ultralytics.json')

        # Load frame paths and filter by gt
        self.frames_paths = glob.glob(join(self.data_path,'AICity/train/S03/c010/vdo', "*." + args.extension))
        self.frames_paths = [path for frame_id,_ in self.gt_bboxes.items() for path in self.frames_paths if frame_id in path]
        self.frames_paths.sort()
        self.data = [{'train':[], 'val':[]}.copy() for _ in range(self.split[1])]

        # OUTPUT PARAMETERS
        self.output_path = args.output_path
        self.save_json = args.save_json
        self.view_img = args.view_img
        self.save_img = args.save_img

        # OF_BM PARAMETERS
        self.window_size = args.window_size
        self.shift = args.shift
        self.stride = args.stride

        # PYFLOW PARAMETERS
        self.alpha = args.alpha, 
        self.ratio = args.ratio, 
        self.minWidth = args.minWidth, 
        self.nOuterFPIterations = args.nOuterFPIterations,
        self.nInnerFPIterations = args.nInnerFPIterations,
        self.nSORIterations = args.nSORIterations,
        self.colType = args.colType

        
    def __len__(self):
        return len(self.frames_paths)

    def train_val_split(self):
        """
        Apply split to specific propotion of the dataset for each strategy (A, B, C).
            A: First 25% frames for training, second 75% for test
            B: K-Fold sorted frames
            C: K-Fold random frames
        """
        if self.split[1] == 1:
            # Strategy A
            if self.split[0] in 'sort':
                self.data[0]['train'] = self.frames_paths[:int(len(self)*.25)]
                self.data[0]['val'] = self.frames_paths[int(len(self)*.25):]

            elif self.split[0] in 'rand':
                train, val, _, _ = train_test_split(np.array(self.frames_paths), np.empty((len(self),)),
                                                    test_size=.75, random_state=0)
                self.data[0]['train'] = train.tolist()
                self.data[0]['val'] = val.tolist()
        else:
            frames_paths = np.array(self.frames_paths)

            shuffle, random_state = False, None
            if self.split[0] in 'rand':
                shuffle, random_state = True, 0

            kf = KFold(n_splits=self.split[1], shuffle=shuffle, random_state=random_state)
            for k, (val_index, train_index) in enumerate(kf.split(frames_paths)):
                self.data[k]['train'] = (frames_paths[train_index]).tolist()
                self.data[k]['val'] = (frames_paths[val_index]).tolist()

    def data_to_model(self):
        to_yolov3(self.data, self.gt_bboxes, self.split[0])
    
    def tracking(self):
        if self.options.tracking_mode in 'overlapping':
            self.det_bboxes = compute_tracking_overlapping(self.det_bboxes, self.frames_paths,
                                                            self.alpha, self.ratio, self.minWidth, 
                                                            self.nOuterFPIterations, self.nInnerFPIterations, 
                                                            self.nSORIterations, self.colType,
                                                            flow_method=self.options.OF_mode)
        elif self.options.tracking_mode in 'kalman':
            self.det_bboxes = compute_tracking_kalman(self.det_bboxes)

    def get_mAP(self, k=None):
        """
        Estimats the mAP using the VOC evaluation

        :return: map of all estimated frames
        """
        if self.mode == 'eval':
            if k is None:
                mAP50 = voc_eval(self.gt_bboxes, self.data[0]['val'], self.det_bboxes)[2]
                mAP70 = voc_eval(self.gt_bboxes, self.data[0]['val'], self.det_bboxes, use_07_metric=True)[2]
            else:
                mAP50 = voc_eval(self.gt_bboxes, self.data[k]['val'], self.det_bboxes)[2]
                mAP70 = voc_eval(self.gt_bboxes, self.data[k]['val'], self.det_bboxes, use_07_metric=True)[2]
        else:
            mAP50 = voc_eval(self.gt_bboxes, self.frames_paths, self.det_bboxes)[2]
            mAP70 = voc_eval(self.gt_bboxes, self.frames_paths, self.det_bboxes, use_07_metric=True)[2]
        return mAP50, mAP70

    def get_mIoU(self):
        """
        Estimates the mIoU

        :return: iou of all estimated frames
        """
        return \
            compute_total_miou(self.gt_bboxes, self.det_bboxes, self.frames_paths)

    def save_results(self, name_json):
        """
        Saves results to a JSON file

        :param name_json: name of the json file
        """
        write_json_file(self.det_bboxes, name_json)

    def visualize(self):
        """
        Creates plots for a given frame and bbox estimation
        """
        visualize_background_iou(self.data[0], None, self.gt_bboxes, self.det_bboxes, self.framework,
                                 self.model, self.options.output_path, self.mode)
    



