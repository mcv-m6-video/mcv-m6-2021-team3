import sys
from os.path import join
import glob
import numpy as np
import cv2
from utils import *
from ai_city import AICity
from cv2 import bgsegm

data_path = '../../data'
output_path = '../outputs'
debug = True
method = 'GMG'


def main(argv):
    if len(argv) > 1:
        task = float(argv[1])
    else:
        task = 3

    if int(task) == 1:
        frames_paths = join(data_path, 'AICity/train/S03/c010/vdo')

        if task == 1.1:
            aicity = AICity(frames_paths, resize_factor=0.5, task=task)
            aicity.create_background_model()
            aicity.get_frames_background()
        elif task == 1.2:
            aicity = AICity(frames_paths, resize_factor=0.2, task=task, rm_noise=True, fill=False, noise_opening=True,alpha=5)
            aicity.create_background_model()
            aicity.get_frames_background()

    elif int(task) == 2:
        pass

    elif int(task) == 3:
        frames_paths = join(data_path, 'AICity/train/S03/c010/vdo')
        aicity = AICity(frames_paths, resize_factor=0.5, split_factor=1,pre_denoise=False,laplacian=True)
        frames = aicity.read_frames()
        if method == 'MOG2':
            backSub = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
            for frame in frames:
                bg_fg_MOG2 = backSub.apply(frame) 
                bg_MOG2 = backSub.getBackgroundImage()
                bg_fg_MOG2[bg_fg_MOG2 != 255] = 0
                cv2.imshow('Background',bg_fg_MOG2)
                cv2.waitKey(100)
        elif method == 'KNN':
            backSub = cv2.createBackgroundSubtractorKNN(detectShadows=False)
            for frame in frames:
                bg_fg_KNN = backSub.apply(frame)
                bg_fg_KNN[bg_fg_KNN != 255] = 0
                cv2.imshow('Background',bg_fg_KNN)
                cv2.waitKey(100)
        elif method == 'GMG':
            backSub = cv2.bgsegm.createBackgroundSubtractorGMG()
            for frame in frames:
                bg_fg_GMG = backSub.apply(frame)
                bg_fg_GMG[bg_fg_GMG != 255] = 0
                cv2.imshow('Background',bg_fg_GMG)
                cv2.waitKey(100)
        elif method == 'LSBP':
            backSub = cv2.bgsegm.createBackgroundSubtractorLSBP()
            for frame in frames:
                bg_fg_LSBP = backSub.apply(frame)
                bg_fg_LSBP[bg_fg_LSBP != 255] = 0
                cv2.imshow('Background',bg_fg_LSBP)
                cv2.waitKey(100)
        

    elif int(task) == 4:
        pass

    else:
        raise NameError

if __name__ == "__main__":
    main(sys.argv)

