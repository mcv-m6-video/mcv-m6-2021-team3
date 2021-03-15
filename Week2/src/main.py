import sys
import os
from os.path import join
import cv2
from utils.ai_city import AICity
import matplotlib.pyplot as plt
from utils.refinement import get_single_objs, filter_noise
from tqdm.auto import tqdm
from utils.metrics import voc_eval
import numpy as np

data_path = '../../data'
output_path = '../outputs'
debug = True
test_mode = False
resize_factor = 0.5
method = 'MOG2'
colorspace = "LAB"

def plot_map_alphas(map,alpha):

        plt.plot(alpha,map)
        plt.xlabel('Alpha')
        plt.ylabel('mAP')
        plt.title('Alpha vs mAP')
        plt.show()

def main(argv):
    if len(argv) > 1:
        task = float(argv[1])
    else:
        task = 3

    os.makedirs('outputs',exist_ok=True)

    if int(task) == 1:
        frames_paths = join(data_path, 'AICity/train/S03/c010/vdo')

        if task == 1.1:
            options = {
                'resize_factor': 0.5,
                'denoise': False,
                'split_factor': 0.25,
                'test_mode': True,
                'colorspace': 'gray',
                'extension': 'png',
                'laplacian': True,
                'median_filter': False,
                'bilateral_filter': False,
                'pre_denoise': False,
                'alpha': 3,
                'rho': 0.5,
                'noise_filter': None,
                'fill': False,
                'apply_road_mask': True,
                'adaptive_model': False,
                'save_img': False,
                'task': task
            }

            aicity = AICity(frames_paths, data_path, options)
            aicity.create_background_model()
            aicity.get_frames_background()
        elif task == 1.2:
            options = {
                'resize_factor': 0.5,
                'denoise': False,
                'split_factor': 0.25,
                'test_mode': False,
                'colorspace': 'gray',
                'extension': 'png',
                'laplacian': True,
                'median_filter': True,
                'bilateral_filter': False,
                'pre_denoise': False,
                'alpha': 3,
                'rho': 0.5,
                'noise_filter': 'morph_filter',
                'fill': True,
                'apply_road_mask': True,
                'adaptive_model': False,
                'return_bboxes': True,
                'save_img': False,
                'task': task
            }

            aicity = AICity(frames_paths, data_path, options)
            aicity.create_background_model()
            aicity.get_frames_background()

    elif int(task) == 2:
        frames_paths = join(data_path, 'AICity/train/S03/c010/vdo')

        if task == 2.1:
            options = {
                'resize_factor': 0.5,
                'denoise': False,
                'split_factor': 0.25,
                'test_mode': True,
                'colorspace': 'gray',
                'extension': 'png',
                'laplacian': False,
                'median_filter': False,
                'bilateral_filter': False,
                'pre_denoise': False,
                'alpha': 3,
                'rho': 0.5,
                'noise_filter': None,
                'fill': False,
                'adaptive_model': False,
                'save_img': False,
                'task': task
            }

            aicity = AICity(frames_paths, data_path, options)
            aicity.create_background_model()
            aicity.get_frames_background()

    elif int(task) == 3:
        options = {
            'resize_factor': 0.5,
            'denoise': False,
            'split_factor': 0.25,
            'test_mode': False,
            'colorspace': 'gray',
            'extension': 'png',
            'laplacian': True,
            'median_filter': True,
            'bilateral_filter': False,
            'pre_denoise': False,
            'alpha': 3,
            'rho': 0.5,
            'noise_filter': ['base',True],
            'fill': False,
            'apply_road_mask': True,
            'adaptive_model': False,
            'return_bboxes': True,
            'save_img': False,
            'apply_rout_mask'
            'task': task
        }

        frames_paths = join(data_path, 'AICity/train/S03/c010/vdo')
        aicity = AICity(frames_paths, data_path, options,bg_model=method)
        aicity.create_background_model()
        aicity.get_frames_background()

        mAP = aicity.get_mAP()

        print('mAP: ', mAP)
            

    elif int(task) == 4:
        os.makedirs('outputs/task_4',exist_ok=True)
        frames_paths = join(data_path, 'AICity/train/S03/c010/vdo')
        alphas = [1.5,2]
        mAP = []

        for alpha in alphas:

            options = {
                'resize_factor': 0.5,
                'denoise': False,
                'split_factor': 0.25,
                'test_mode': False,
                'colorspace': 'LAB',
                'extension': 'png',
                'laplacian': False,
                'median_filter': True,
                'bilateral_filter': False,
                'pre_denoise': False,
                'alpha': alpha,
                'rho': 0.05,
                'noise_filter': ['base',True],#'morph_filter',
                'fill': False,
                'apply_road_mask': True,
                'adaptive_model': False,
                'return_bboxes': True,
                'save_img': False,
                'task': task
            }

            aicity = AICity(frames_paths, data_path, options)
            aicity.create_background_model()
            aicity.get_frames_background()

            mAP.append(aicity.get_mAP())

            print('mAP: ', mAP)

        plot_map_alphas(mAP,alphas)

        #aicity.save_results('LAB.json')



    else:
        raise NameError


if __name__ == "__main__":
    main(sys.argv)
