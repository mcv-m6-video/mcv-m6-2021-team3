import os
from os.path import join
from utils.ai_city import AICity
from utils.kitti import KITTI
from config.config import Config

def main(args):

    kitti = KITTI(args.data_path, args.OF_mode, args)

    kitti.estimate_OF()
    msen, pepn = kitti.get_MSEN_PEPN()
    kitti.visualize()
    print('MSEN:', msen)
    print('PEPN:', pepn)
    

if __name__ == "__main__":
    main(Config().get_args())