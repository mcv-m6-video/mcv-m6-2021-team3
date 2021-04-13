import os
from os.path import join
import glob
from datasets.ai_city import AICity
from config.config import Config
from utils.utils import read_video_file

def main(args):

    aicity = AICity(args)
    aicity.track(args.seqs)

    #print(args.data_path)

    '''for path in glob.glob(join(args.data_path,'AICity/train/S04/*/*.avi')):
        read_video_file(path)'''

if __name__ == "__main__":
    main(Config().get_args())
