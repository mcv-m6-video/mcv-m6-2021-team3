import sys
import os
from os.path import join
import cv2
from utils.ai_city import AICity
import matplotlib.pyplot as plt

data_path = '../../data'
output_path = '../outputs'
method = 'GAUSSIAN-MASK-RCNN'


def plot_map_alphas(map, alpha):
    plt.plot(alpha, map)
    plt.xlabel('Alpha')
    plt.ylabel('mAP')
    plt.title('Alpha vs mAP')
    plt.show()


def main(argv):
    if len(argv) > 1:
        task = float(argv[1])
    else:
        task = 3

    os.makedirs('outputs', exist_ok=True)

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
                'return_bboxes': True,
                'task': task
            }

            aicity = AICity(frames_paths, data_path, options)
            aicity.create_background_model()
            aicity.get_frames_background()

    elif int(task) == 3:
        options = {
            'resize_factor': 1,
            'denoise': False,
            'split_factor': 0.25,
            'test_mode': False,
            'colorspace': 'gray',
            'extension': 'png',
            'laplacian': False,
            'median_filter': False,
            'bilateral_filter': False,
            'pre_denoise': False,
            'alpha': 5,
            'rho': 0.1,
            'noise_filter': None,
            'fill': False,
            'adaptive_model': True,
            'save_img': False,
            'return_bboxes': True,
            'task': task
        }

        frames_paths = join(data_path, 'AICity/train/S03/c010/vdo')
        aicity = AICity(frames_paths, data_path, options)
        frames = aicity.read_frames()

        if method == 'MOG2':
            backSub = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
            # bg_MOG2 = backSub.getBackgroundImage()

        elif method == 'KNN':
            backSub = cv2.createBackgroundSubtractorKNN(detectShadows=False)

        elif method == 'GMG':
            backSub = cv2.bgsegm.createBackgroundSubtractorGMG()

        elif method == 'LSBP':
            backSub = cv2.bgsegm.createBackgroundSubtractorLSBP()

        if method in ['MOG2', 'KNN', 'GMG', 'LSBP']:
            for frame in frames:
                bg = backSub.apply(frame)
                bg[bg != 255] = 0
                cv2.imshow('Background', bg)
                cv2.waitKey(100)

        # if method == "GAUSSIAN-MASK-RCNN":
        #     frames_paths = join(data_path, 'AICity/train/S03/c010/rcnn-masks')
        #     aicity = AICity(frames_paths, data_path, options)
        #     aicity.create_background_model()
        #     aicity.get_frames_background()
        #     print(aicity.get_mAP())

    elif int(task) == 4:
        os.makedirs('outputs/task_4', exist_ok=True)
        frames_paths = join(data_path, 'AICity/train/S03/c010/vdo')
        alphas = [1.5]
        mAP = []

        for alpha in alphas:
            options = {
                'resize_factor': 0.5,
                'denoise': True,
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
                'noise_filter': ['base', True],  # 'morph_filter',
                'fill': True,
                'adaptive_model': True,
                'return_bboxes': True,
                'save_img': True,
                'task': task
            }

            aicity = AICity(frames_paths, data_path, options)
            aicity.create_background_model()
            aicity.get_frames_background()

            mAP.append(aicity.get_mAP())

            print('mAP: ', mAP)

        plot_map_alphas(mAP, alphas)

        aicity.save_results('LAB.json')

    else:
        raise NameError


if __name__ == "__main__":
    main(sys.argv)
