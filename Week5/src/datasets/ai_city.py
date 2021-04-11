from os.path import join, basename, exists
import glob
from tqdm import tqdm
import random

import matplotlib.pyplot as plt
import matplotlib.patches as patches
#matplotlib inline
from IPython import display as dp
from termcolor import colored

import numpy as np
import matplotlib.pyplot as plt
import json
import cv2

import numpy as np
from skimage import io
import os
import time

from datasets.load_seq import LoadSeq
from utils.utils import write_json_file, read_json_file, write_yaml_file

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

        # OUTPUT PARAMETERS
        self.output_path = args.output_path

        # DETECTOR PARAMETERS
        self.det_params = dict(
            mode = args.mode,
            model = args.model,
            weights = args.weights,
            hyp = args.hyp,
            data = args.data_yolov3,
            epochs = args.epochs,
            batch_size = args.batch_size,
            img_size = args.img_size,
            conf_thres = args.conf_thres,
            iou_thres = args.iou_thres,
            name = args.model+'_'.join(args.seq_train)
        )
        self.seq_train = args.seq_train
        self.seq_test = args.seq_test

        # LOAD SEQUENCE
        self.sequences = {}
        for seq in os.listdir(self.data_path):
            if 'S' in seq[0]:
                det_name = '_'.join((self.model, self.framework+'.json'))
                self.sequences.update({seq:LoadSeq(self.data_path, seq, self.output_path, det_name, det_params=self.det_params)})
        
    def __len__(self):
        return len(self.sequences)

    def data_to_model(self, save_path='data/yolov3_finetune'):
        save_path = join(save_path,'-'.join(self.seq_train)+'_'+'-'.join(self.seq_test))
        os.makedirs(save_path,exist_ok=True)
        
        if len(os.listdir(save_path)) == 4:
            print(colored('Your data is already in the appropriate format! :)', 'green'))
            return
        
        print('Preparing data...')
        files_txt = {'train':[],'val':[],'test':[]}
        for seq, sequence in self.sequences.items():
            if seq in self.seq_train:
                [files_txt[split].append(paths) for split, paths in sequence.data_to_model(mode='train').items()]
            else:
                [files_txt[split].append(paths) for split, paths in sequence.data_to_model(mode='test').items()]
        
        yaml_dict = dict(
            nc = 1,
            names = ['car']
        )
        for split, paths in files_txt.items():
            paths = [path for cam in paths for path in cam]
            file_out = open(join(save_path,split+'.txt'), 'w')
            file_out.writelines(paths)
            yaml_dict.update({split:join(save_path,split+'.txt')})

        write_yaml_file(yaml_dict,join(save_path,'cars.yaml'))
        print(colored('DONE!', 'green'))       

    def detect_on_seq(self, seqs):
        for seq in seqs:
            print(self.det_params['mode']+f' on sequence: {seq}')
            mAP50, mAP70 = self.sequences[seq].detect()
            print(f'Sequence: {seq}, mAP50={mAP50}, mAP70={mAP70}')
    


