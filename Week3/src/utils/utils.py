from os import makedirs
from os.path import exists, join
import glob
import numpy as np
import pathlib
import tqdm
import random
import json
from termcolor import colored
import imageio
import cv2
import subprocess

def write_json_file(data_dict, json_file):
    """
    Write json file
    :param data_dict: dict containing data to store
    :param json_file: name of the json file
    """

    with(open(json_file, 'w')) as f:
        json.dump(data_dict, f)

    if exists(json_file):
        print('Json file ' + colored('\'' + json_file + '\'', 'blue') + ' written successfully!')


def read_json_file(json_file):
    """
    Read json file
    :param json_file: name of the json file
    :return: dictionary with the json data
    """

    with(open(json_file, 'r+')) as f:
        data = json.load(f)

    if data is not None:
        print('Json file ' + colored('\'' + json_file + '\'', 'blue') + ' loaded successfully!')

    return data


def read_video_file(video_file):
    """
    Read video from file
    :param video_file: name of the video file
    """

    path = pathlib.Path(video_file)
    folder = join(str(path.parent), path.name.split('.')[0])
    makedirs(folder, exist_ok=True)

    capture = cv2.VideoCapture(video_file)
    n_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    counter = 0
    progress_bar = tqdm.tqdm(range(n_frames), total=n_frames)

    while capture.isOpened():
        retrieved, frame = capture.read()

        if retrieved:
            cv2.imwrite(join(folder, str(counter).zfill(4) + '.png'), frame)
            counter += 1
            progress_bar.update(1)
        else:
            print("End of video")
            break

    capture.release()


def dict_to_list(frame_info, tlwh=True):
    """
    Transform a dictionary into a list
    :param frame_info: dictionary with the information needed to create the list
    :param tlwh: Boolean that determine if the format is top-left-width-height
    :return: return the list created
    """

    if tlwh:
        return [[obj['bbox'][0],
                 obj['bbox'][1],
                 obj['bbox'][2] - obj['bbox'][0],
                 obj['bbox'][3] - obj['bbox'][1]] for obj in frame_info]
    else:
        return [[obj['bbox'][0], obj['bbox'][1], obj['bbox'][2], obj['bbox'][3]] for obj in frame_info]


def frames_to_gif(save_dir, ext):
    img_paths = glob.glob(join(save_dir, '*.' + ext))
    img_paths.sort()
    gif_dir = save_dir + '.gif'
    with imageio.get_writer(gif_dir, mode='I') as writer:
        for img_path in img_paths:
            image = imageio.imread(img_path)
            writer.append_data(image)

    print('Gif saved at ' + gif_dir)

def get_weights(model, framework):
    makedirs('data/weights/',exist_ok=True)
    if framework in 'detectron2':
        model_path = 'data/weights/'+model+'.pkl'
    elif framework in 'ultralytics':
        model_path = 'data/weights/'+model+'.pt'
        
    if not exists(model_path):
        subprocess.call(['sh','./data/scripts/get_'+model+'.sh'])
    return model_path

def frame_id(id):
    return ('%04d' % id)
