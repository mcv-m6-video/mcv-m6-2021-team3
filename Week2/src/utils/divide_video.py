from os.path import join
import pathlib
from os import makedirs
import tqdm
import cv2

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


read_video_file('../../../data/AICity/train/S03/c010/vdo.avi')