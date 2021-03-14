import sys
from os.path import join
import cv2
from utils.ai_city import AICity

data_path = '../../data'
output_path = '../outputs'
debug = True
test_mode = False
resize_factor = 0.5
method = 'GMG'
colorspace = "LAB"


def main(argv):
    if len(argv) > 1:
        task = float(argv[1])
    else:
        task = 1.2


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
                'task': task
            }

            aicity = AICity(frames_paths, options)
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
                'task': task
            }

            aicity = AICity(frames_paths, options)
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
                'task': task
            }

            aicity = AICity(frames_paths, options)
            aicity.create_background_model()
            aicity.get_frames_background()

    elif int(task) == 3:
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
            'task': task
        }

        frames_paths = join(data_path, 'AICity/train/S03/c010/vdo')
        aicity = AICity(frames_paths, options)
        frames = aicity.read_frames()

        if method == 'MOG2':
            backSub = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
            # bg_MOG2 = backSub.getBackgroundImage()
            s
        elif method == 'KNN':
            backSub = cv2.createBackgroundSubtractorKNN(detectShadows=False)

        elif method == 'GMG':
            backSub = cv2.bgsegm.createBackgroundSubtractorGMG()

        elif method == 'LSBP':
            backSub = cv2.bgsegm.createBackgroundSubtractorLSBP()

        for frame in frames:
            bg_fg_KNN = backSub.apply(frame)
            bg_fg_KNN[bg_fg_KNN != 255] = 0
            cv2.imshow('Background', bg_fg_KNN)
            cv2.waitKey(100)

    elif int(task) == 4:
        frames_paths = join(data_path, 'AICity/train/S03/c010/vdo')

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
            'task': task
        }

        aicity = AICity(frames_paths, options)
        aicity.create_background_model()
        aicity.get_frames_background()

    else:
        raise NameError


if __name__ == "__main__":
    main(sys.argv)
