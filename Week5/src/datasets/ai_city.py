from os.path import join, basename, exists
import glob
from tqdm import tqdm
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

from datasets.load_seq import LoadSeq

from utils.metrics import voc_eval, compute_iou, compute_centroid, compute_total_miou, interpolate_bb
from utils.utils import write_json_file, read_json_file, str_frame_id, dict_to_list_IDF1, dict_to_list_track, update_data
from utils.visualize import visualize_background_iou

from models.yolov3 import UltralyricsYolo, to_yolov3
#from models.tracking import compute_tracking_overlapping, compute_tracking_kalman

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
        self.data_path = join(args.data_path,'AICity','train')
        self.img_size = args.img_size
        self.split = args.split
        self.model = args.model
        self.framework = args.framework
        self.mode = args.mode

        # OUTPUT PARAMETERS
        self.output_path = args.output_path

        # LOAD SEQUENCE
        self.sequences = {}
        for seq in os.listdir(self.data_path):
            if 'S' in seq[0]:
                det_name = '_'.join((self.model, self.framework, , +'.json'))
                self.sequences.update({seq:LoadSeq(self.data_path, seq, self.output_path, det_name)})

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

    def data_to_model(self):
        to_yolov3(self.data, self.gt_bboxes, self.split[0])
    
    def inference(self, weights=None):
        model = UltralyricsYolo(weights, args=self.options)
        
        self.frames_paths = self.frames_paths[int(len(self.frames_paths)*0.25):]

        for file_name in tqdm(self.frames_paths, 'Model predictions ({}, {})'.format(self.model, self.framework)):
            pred = model.predict(file_name)
            frame_id = file_name[-8:-4]
            for (bbox), conf in pred:
                self.det_bboxes = update_data(self.det_bboxes, frame_id, *bbox, conf)
        
        if self.save_json:
            save_path = join(self.options.output_path, self.mode+'/')
            os.makedirs(save_path, exist_ok=True)


            if self.options.mode == 'train':
                write_json_file(self.det_bboxes,save_path+'_'.join((self.model, self.framework+'_training.json')))
            else:
                write_json_file(self.det_bboxes,save_path+'_'.join((self.model, self.framework+'.json')))

            if self.mode == 'inference':
                write_json_file(self.det_bboxes,save_path+'_'.join((self.model, self.framework+'.json')))
            else:
                write_json_file(self.det_bboxes,join(self.options.output_path,self.mode+'/') +'_'.join((self.model, self.framework, self.split[0], str(self.options.conf_thres), str(self.options.iou_thres) +'.json')))#save_path+'_'.join((self.model, self.framework, self.split[0]+'.json')))


    
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
    



