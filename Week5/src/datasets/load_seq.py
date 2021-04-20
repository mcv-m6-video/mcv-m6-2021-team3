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
from termcolor import colored


from config.config_multitracking import ConfigMultiTracking
from modes.ultralytics_yolo import UltralyricsYolo, to_yolov3

#from modes.tf_models import TFModel, to_tf_record
from modes.multitracking import iamai_multitracking
from modes.tracking import compute_tracking_overlapping, compute_tracking_kalman, compute_tracking_iou
from utils.visualize import visualize_trajectories, visualize_filter_roi
from utils.utils import write_json_file, read_json_file, update_data, match_trajectories, dist_to_roi, filter_by_roi, read_txt_to_dict
from utils.metrics import voc_eval, compute_iou, compute_total_miou, interpolate_bb, IDF1, compute_IDmetrics, compute_IDmetrics_multi

import motmetrics as mm


import matplotlib.pyplot as plt

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
        update_data(annot, frame_id - 1, xmin, ymin, xmin + width, ymin + height, conf, int(bb_id))
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
    def __init__(self, data_path, seq, output_path, tracking_mode, det_name, OF_mode, extension='jpg', det_params=None):
        """
        Init of the Load Sequence class

        :param data_path: path to data
        :param args: configuration for the current estimation
        """

        # INPUT PARAMETERS
        self.data_path = data_path
        self.seq = seq
        self.det_params = det_params
        self.det_name = self.det_params['mode'] + '2_' + det_name
        if self.det_params['mode'] == 'tracking':
            self.det_name = 'eval2_' + det_name
        self.track_mode = tracking_mode
        self.OF_mode = OF_mode
        self.mt_args = ConfigMultiTracking().get_args()

        # OUTPUT PARAMETERS
        self.output_path = output_path

        # Load detections and load frame paths and filter by gt
        self.gt_bboxes = {}
        self.det_bboxes = {}
        self.frames_paths = {}
        self.tracker = {}
        self.mask = {}

        self.accumulators = {}

        if self.mt_args.mode in 'mtmc_vt':
            if seq not in self.mt_args.txt_name:
                pass
            else:
                self.det_bboxes = read_txt_to_dict(self.mt_args.txt_name)

                for cam, _ in self.det_bboxes.items():
                    # Save paths to frames
                    cam_paths = glob.glob(join(data_path,seq,cam,'vdo/*.'+extension))
                    #cam_paths = [path for frame_id,_ in self.gt_bboxes[cam].items() for path in cam_paths if frame_id in path]
                    cam_paths.sort()
                    self.frames_paths.update({cam:cam_paths})
                    # Load gt
                    self.gt_bboxes.update({cam: load_annot(join(data_path, seq, cam), 'gt/gt.txt')})
                    # Creat accumulator
                    self.accumulators.update({cam: mm.MOTAccumulator()})

        else:
            for cam in os.listdir(join(data_path, seq)):
                if '.' in cam:
                    continue
                # Save paths to frames
                cam_paths = glob.glob(join(data_path,seq,cam,'vdo/*.'+extension))
                #cam_paths = [path for frame_id,_ in self.gt_bboxes[cam].items() for path in cam_paths if frame_id in path]
                cam_paths.sort()
                self.frames_paths.update({cam:cam_paths})
                # Load cam mask (roi)
                self.mask.update({cam:dist_to_roi(join(data_path,seq,cam,'roi.jpg'))})
                
                # Load gt
                self.gt_bboxes.update({cam: load_annot(join(data_path, seq, cam), 'gt/gt.txt')})

                # Check if detections already computed
                json_path = join(output_path, seq, cam)
                os.makedirs(json_path, exist_ok=True)
                json_path = join(json_path, self.det_name)
                if exists(json_path):
                    self.det_bboxes.update({cam:read_json_file(json_path)})
                    for frame_id in self.frames_paths[cam]:
                        idx = frame_id[-8:-4]
                        if idx not in self.det_bboxes[cam].keys():
                            self.det_bboxes[cam] = update_data(self.det_bboxes[cam], idx,*[-1,-1,-1,-1],0,0, True)
                else:
                    self.det_bboxes.update({cam:{}})
                
                # Creat accumulator 
                self.accumulators.update({cam: mm.MOTAccumulator()})

    def train_val_split(self, split=.25, mode='test'):
        """
        Apply split to specific propotion of the dataset.
        """
        self.data = {'train':{},'val':{},'test':{}}
        if mode in 'train':
            # Define cams used to train and validate
            cams = self.frames_paths.keys()
            cams_val = random.sample(cams, int(len(cams) * split))
            cams_train = list(set(cams) - set(cams_val))

            self.data.update({'train': dict(filter(lambda cam: cam[0] in cams_train, self.frames_paths.items()))})
            self.data.update({'val': dict(filter(lambda cam: cam[0] in cams_val, self.frames_paths.items()))})

        else:
            # The whole sequence used to test
            self.data.update({'test':self.frames_paths})
    
    def data_to_model(self, split=.25, mode='test', writer=None):
        self.train_val_split(split, mode)
        if self.det_params['framework'] in 'ultralytics':
            return to_yolov3(self.data, self.gt_bboxes)
        elif self.det_params['framework'] in 'tf_models':
            return to_tf_record(self.data, self.gt_bboxes, writer)
    
    def detect(self):

        if self.det_params['framework'] in 'ultralytics':
            model = UltralyricsYolo(self.det_params['weights'], args=self.det_params)
        elif self.det_params['framework'] in 'tf_models':
            model = TFModel(self.det_params['model'],self.det_params['weights'],self.det_params['iou_thres'], self.det_params['coco_model'])

        print(self.det_params['mode']+f' for sequence: {self.seq}')

        for cam, paths in self.frames_paths.items():
            if len(self.det_bboxes[cam]) > 0:
                continue
            for file_name in tqdm(paths, 'Model predictions ({}, {})'.format(cam, self.det_params['model'])):
                pred = model.predict(file_name)
                frame_id = file_name[-8:-4]
                if len(pred) == 0:
                    self.det_bboxes[cam] = update_data(self.det_bboxes[cam], frame_id, *[-1, -1, -1, -1], 0.0)
                for (bbox), conf in pred:
                    self.det_bboxes[cam] = update_data(self.det_bboxes[cam], frame_id, *bbox, conf)

            write_json_file(self.det_bboxes[cam], join(self.output_path, self.seq, cam, self.det_name))

        return self.get_mAP()

    def single_cam_tracking(self):
        for cam, det_bboxes in self.det_bboxes.items():
            det_bboxes = filter_by_roi(det_bboxes,self.mask[cam])
            if self.track_mode in ['overlapping', 'kalman']:

                if self.track_mode in 'overlapping':            
                    det_bboxes = compute_tracking_overlapping(det_bboxes, self.frames_paths[cam], flow_method= self.OF_mode)

                elif self.track_mode in 'kalman':
                    det_bboxes = compute_tracking_kalman(det_bboxes, self.gt_bboxes[cam])#, self.accumulators[cam])

                self.ID_metrics.update({cam:compute_IDmetrics(self.gt_bboxes[cam],det_bboxes,self.accumulators[cam],self.frames_paths[cam][0])})
                print(f'Camera: {cam}')
                print(self.ID_metrics[cam])

                self.det_bboxes[cam] = det_bboxes

            elif self.track_mode in 'iou_track':
                self.tracker.update({cam:compute_tracking_iou(det_bboxes,cam,self.data_path)})

    def tracking(self, multitracking):
        self.ID_metrics={}
        if multitracking:
            if self.mt_args.mode in 'color_hist':
                self.single_cam_tracking()
                hist_multitracking(self.det_bboxes, self.frames_paths)
            elif self.mt_args.mode in 'iamai':
                iamai_multitracking(self.mt_args)
            elif self.mt_args.mode in 'mtmc_vt':
                self.ID_metrics = compute_IDmetrics_multi(self.gt_bboxes,self.det_bboxes,self.accumulators,self.frames_paths)
                print(colored('IDF1: ', 'blue', attrs=['bold']) + str(self.ID_metrics.idf1.OVERALL))
                print(colored('IDP: ', 'blue', attrs=['bold']) + str(self.ID_metrics.idp.OVERALL))
                print(colored('IDR: ', 'blue', attrs=['bold']) + str(self.ID_metrics.idr.OVERALL))
                print(colored('Precision: ', 'blue', attrs=['bold']) + str(self.ID_metrics.precision.OVERALL))
                print(colored('Recall: ', 'blue', attrs=['bold']) + str(self.ID_metrics.recall.OVERALL))
        else:
            self.single_cam_tracking()
          
    def get_mAP(self):
        """
        Estimats the mAP using the VOC evaluation

        :return: map of all estimated frames
        """
        mAP50, mAP70 = [], []
        print(self.det_params['mode'] + f' mAP for sequence: {self.seq}')
        for gt_bboxes, cam_paths, det_bboxes in zip(self.gt_bboxes.values(), self.frames_paths.values(),
                                                    self.det_bboxes.values()):
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
        #self.visualize_filter()
        for (cam,cam_paths), det_bboxes in tqdm(zip(self.frames_paths.items(), self.det_bboxes.values()), 'Saving tracking qualitative results'):
            path_in = dirname(cam_paths[0])
            if self.det_params['mode'] == 'tracking':
                visualize_trajectories(path_in, join(self.output_path,self.seq,cam), det_bboxes)
    
    def visualize_filter(self):
        for cam, det_bboxes in self.det_bboxes.items():
            det_bboxes_filter = filter_by_roi(det_bboxes,self.mask[cam])
            visualize_filter_roi(self.frames_paths[cam],self.gt_bboxes[cam], det_bboxes, det_bboxes_filter,
                                 self.mask[cam], join(self.output_path,self.seq,cam))

