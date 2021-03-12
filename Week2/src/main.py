import sys
from os.path import join
import glob
import numpy as np
import cv2
from utils import *

data_path = '../../data'
output_path = '../outputs'
debug = True


def main(argv):
    if len(argv) > 1:
        task = float(argv[1])
    else:
        task = 1.1

    if int(task) == 1:
        frames_paths = glob.glob(join(join(data_path, 'AICity/train/S03/c010/vdo'), '*.png'))
        frames_paths.sort()

        bg_modeling_frames_paths = frames_paths[:int(len(frames_paths)*0.25)]  # 535 frames
        bg_frames_paths = frames_paths[int(len(frames_paths)*0.25):]  # 1606 frames

        if debug:
            print(len(bg_modeling_frames_paths), len(bg_frames_paths))

        bg_modeling_frames = read_frames(bg_modeling_frames_paths)
        gaussian_model = model_background(bg_modeling_frames)
        bg_modeling_frames = None
        bg_frames = read_frames(bg_frames_paths)
        
        for idx, frame in enumerate(bg_frames):
            print(idx)
            if task == 1.1:
                bg = get_frame_background(frame, gaussian_model)
            elif task == 1.2:
                bg = get_frame_background(frame, gaussian_model, rm_noise=True)

            cv2.imshow("Background of frame", cv2.resize(bg, (int(1920*0.5), int(1080*0.5))))
            cv2.imshow("Real", cv2.resize(frame, (int(1920 * 0.5), int(1080 * 0.5))))
            cv2.waitKey(100)





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

