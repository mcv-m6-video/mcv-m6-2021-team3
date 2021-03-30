import cv2
import png
import glob
import os
import numpy as np
from os.path import join, exists
from models.stabilize import seq_stabilization_BM, seq_stabilization_LK

class LoadSeq():
    def __init__(self, data_path, args):
        """
        Init of the Load Sequence class

        :param data_path: path to data
        :param args: configuration for the current estimation
        """
        
        # INPUT PARAMETERS
        self.data_path = data_path
        self.seq_path = args.seq_path
        self.output_path = join(args.output_path, args.seq_path)
        os.makedirs(self.output_path, exist_ok=True)        

        # Stabilization BM
        self.frames_paths = glob.glob(join(self.data_path,self.seq_path,"*." + args.extension))
        self.frames_paths.sort()

        # Stabilization mode
        self.stabilization = args.modelStab
        # For block matching estimation
        self.block_matching = dict(
            window_size = args.window_size,
            shift = args.shift,
            stride = args.stride)
    
    def stabilize_seq(self):
        if self.stabilization in 'ours':
            seq_stabilization_BM(self.frames_paths, self.output_path, self.block_matching)
        if self.stabilization in 'opencv2':
            seq_stabilization_LK(self.frames_paths, self.output_path)
