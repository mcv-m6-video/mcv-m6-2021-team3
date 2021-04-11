import os
from os.path import join
import glob
from datasets.ai_city import AICity
from config.config import Config

def main(args):

    aicity = AICity(args)
    aicity.data_to_model()
    #aicity.detect_on_seq(['S01','S03','S04'])

    #print(args.data_path)

    '''for path in glob.glob(join(args.data_path,'AICity/train/*/*/*.avi')):
        read_video_file(path)'''

if __name__ == "__main__":
    main(Config().get_args())
