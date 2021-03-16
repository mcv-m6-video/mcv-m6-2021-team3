import sys
import os
from os.path import join
import cv2
from utils.ai_city import AICity
from utils.visualize import plot_map_alphas
from utils.utils import write_json_file
import matplotlib.pyplot as plt

data_path = '../../data'
output_path = '../outputs'
debug = True
test_mode = False
resize_factor = 0.5
method = 'MOG2'
colorspace = "LAB"

def main(argv):
    if len(argv) > 1:
        task = float(argv[1])
    else:
        task = 1.1

    os.makedirs('outputs',exist_ok=True)

    if int(task) == 1:
        frames_paths = join(data_path, 'AICity/train/S03/c010/vdo')

        if task == 1.1:
            alphas = [1,1.5,2,3,4,5]
            mAP = []
            for alpha in alphas:
                options = {
                    'resize_factor': 0.5,
                    'split_factor': 0.25,
                    'test_mode': False,
                    'colorspace': 'gray',
                    'extension': 'png',
                    'laplacian': False,
                    'median_filter': False,
                    'bilateral_filter': False,
                    'pre_denoise': False,
                    'alpha': alpha,
                    'rho': 0.5,
                    'noise_filter': None,
                    'fill': False,
                    'apply_road_mask': True,
                    'adaptive_model': False,
                    'save_img': False,
                    'visualize': True,
                    'return_bboxes': True,
                    'task': task
                }

                aicity = AICity(frames_paths, data_path, options)
                aicity.create_background_model()
                aicity.get_frames_background()

                mAP.append(aicity.get_mAP())

                write_json_file({'miou':aicity.miou, 'std_miou':aicity.std_iou},str(alpha)+'.json')

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
                'task': int(task)
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
                'task': int(task)
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
            'noise_filter': ['morph_filter', True],
            'fill': False,
            'apply_road_mask': True,
            'adaptive_model': False,
            'return_bboxes': True,
            'save_img': False,
            'task': int(task)
        }

        methods = ['MOG2', 'KNN', 'GMG', 'LSBP']
        maps = []

        for method in methods:
            frames_paths = join(data_path, 'AICity/train/S03/c010/vdo')
            aicity = AICity(frames_paths, data_path, options, bg_model=method)
            aicity.create_background_model()
            aicity.get_frames_background()

            maps.append(aicity.get_mAP())

    elif int(task) == 4:
        os.makedirs('outputs/task_4',exist_ok=True)
        frames_paths = join(data_path, 'AICity/train/S03/c010/vdo')

        options = {
            'resize_factor': 0.5,
            'denoise': False,
            'split_factor': 0.25,
            'test_mode': False,
            'colorspace': 'YCbCr',
            'extension': 'png',
            'laplacian': False,
            'median_filter': True,
            'bilateral_filter': False,
            'pre_denoise': False,
            'alpha': 1.75,
            'rho': 0.001,
            'noise_filter': ['base', False],  # 'morph_filter',
            'fill': True,
            'apply_road_mask': True,
            'adaptive_model': False,
            'return_bboxes': True,
            'save_img': False,
            'visualize': True,
            'task': int(task)
        }

        aicity = AICity(frames_paths, data_path, options)
        aicity.create_background_model()
        aicity.get_frames_background()

        mAP  = aicity.get_mAP()

        print('mAP: ', mAP)

        #plot_map_alphas(mAP,alphas)

        #aicity.save_results('LAB.json')

    else:
        raise NameError


if __name__ == "__main__":
    main(sys.argv)
