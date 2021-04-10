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

    def detect_on_seq(self, seqs):
        for seq in seqs:
            print(self.det_params['mode']+f' on sequence: {seq}')
            mAP50, mAP70 = self.sequences[seq].detect()
            print(f'Sequence: {seq}, mAP50={mAP50}, mAP70={mAP70}')
    



