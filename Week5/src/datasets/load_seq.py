import cv2
import png
import glob
import os
import numpy as np
from os.path import join, exists
from sklearn.model_selection import train_test_split, KFold
import xml.etree.ElementTree as ET

from utils.utils import read_json_file, update_data

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
    def __init__(self, data_path, seq, output_path, det_name, extension='png'):
        """
        Init of the Load Sequence class

        :param data_path: path to data
        :param args: configuration for the current estimation
        """
        
        # INPUT PARAMETERS
        self.data_path = data_path
        self.seq = seq

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
            

            # Save paths to frames
            cam_paths = glob.glob(join(data_path,seq,cam,'vdo/*.'+extension))
            cam_paths = [path for frame_id,_ in self.gt_bboxes[cam].items() for path in cam_paths if frame_id in path]
            cam_paths.sort()
            self.frames_paths.update({cam:cam_paths})
        

    def train_val_split(self, split):
        """
        Apply split to specific propotion of the dataset for each strategy (A, B, C).
            A: First 25% frames for training, second 75% for test
            B: K-Fold sorted frames
            C: K-Fold random frames
        """
        self.data = [{'train':[], 'val':[]}.copy() for _ in range(split[1])]
        if split[1] == 1:
            # Strategy A
            if split[0] in 'sort':
                self.data[0]['train'] = self.frames_paths[:int(len(self)*.25)]
                self.data[0]['val'] = self.frames_paths[int(len(self)*.25):]

            elif split[0] in 'rand':
                train, val, _, _ = train_test_split(np.array(self.frames_paths), np.empty((len(self),)),
                                                    test_size=.75, random_state=0)
                self.data[0]['train'] = train.tolist()
                self.data[0]['val'] = val.tolist()
        else:
            frames_paths = np.array(self.frames_paths)

            shuffle, random_state = False, None
            if split[0] in 'rand':
                shuffle, random_state = True, 0

            kf = KFold(n_splits=split[1], shuffle=shuffle, random_state=random_state)
            for k, (val_index, train_index) in enumerate(kf.split(frames_paths)):
                self.data[k]['train'] = (frames_paths[train_index]).tolist()
                self.data[k]['val'] = (frames_paths[val_index]).tolist()
