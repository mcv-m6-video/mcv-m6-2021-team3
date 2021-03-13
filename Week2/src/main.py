import sys
from os.path import join
import glob
import numpy as np
import cv2
from ai_city import AICity

data_path = '../../data'
output_path = '../outputs'
debug = True
test_mode = False

def main(argv):
    if len(argv) > 1:
        task = float(argv[1])
    else:
        task = 1.2

    if int(task) == 1:
        frames_paths = join(data_path, 'AICity/train/S03/c010/vdo')

        if task == 1.1:
            aicity = AICity(frames_paths, resize_factor=0.5, task=task, test_mode = test_mode)
            aicity.create_background_model()
            aicity.get_frames_background()
        elif task == 1.2:
            aicity = AICity(frames_paths, resize_factor=0.5, task=task, test_mode = test_mode,  
                            rm_noise=True, fill=True, noise_opening=True, pre_denoise=True, laplacian = True, alpha=5)
            aicity.create_background_model()
            aicity.get_frames_background()

    elif int(task) == 2:
        pass

    elif int(task) == 3:
        pass

    elif int(task) == 4:
        pass

    else:
        raise NameError

if __name__ == "__main__":
    main(sys.argv)

