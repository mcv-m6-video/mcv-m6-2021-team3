import os
from os.path import join
from utils.ai_city import AICity
from utils.kitti import KITTI
from config.config import Config

def main(args):

    os.makedirs(join(args.output_path, str(args.task)), exist_ok=True)
    kitti = KITTI(args.data_path, args.OF_mode, args)

    kitti.estimate_OF()
    

if __name__ == "__main__":
    main(Config().get_args())