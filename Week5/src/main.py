import os
from os.path import join
import glob
#from ai_city import AICity
from utils.utils import write_json_file, dict_to_list_IDF1, read_video_file
from config.config import Config
#from utils.yolov3 import UltralyricsYolo
from utils.visualize import visualize_trajectories, plot_idf1_thr
from utils.metrics import IDF1


def main(args):

    print(args.data_path)

    for path in glob.glob(join(args.data_path,'AICity/train/*/*/*.avi')):
        read_video_file(path)

if __name__ == "__main__":
    main(Config().get_args())
