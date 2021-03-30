import os
from os.path import join
from datasets.ai_city import AICity
from datasets.kitti import KITTI
from datasets.load_seq import LoadSeq
from config.config import Config

def main(args):

    # task 1
    #kitti = KITTI(args.data_path, args.OF_mode, args)
    #kitti.estimate_OF()
    #msen, pepn = kitti.get_MSEN_PEPN()
    #kitti.visualize()
    #print('MSEN:', msen)
    #print('PEPN:', pepn)

    # task 2
    #seq = LoadSeq(args.data_path, args)
    #seq.stabilize_seq()

    #task 3
    ai_city = AICity(args)
    det_bboxes = ai_city.tracking()

if __name__ == "__main__":
    main(Config().get_args())