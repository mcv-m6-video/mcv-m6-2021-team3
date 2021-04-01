import os
from os.path import join
from datasets.ai_city import AICity
from datasets.kitti import KITTI
from datasets.load_seq import LoadSeq
from config.config import Config
from utils.utils import dict_to_list_IDF1
from utils.visualize import visualize_trajectories
from utils.metrics import IDF1

def main(args):

    # task 1
    # kitti = KITTI(args.data_path, args.OF_mode, args)
    # kitti.estimate_OF()
    # msen, pepn = kitti.get_MSEN_PEPN()
    # kitti.visualize()
    # print('MSEN:', msen)
    # print('PEPN:', pepn)

    # task 2
    #seq = LoadSeq(args.data_path, args)
    #seq.stabilize_seq()

    #task 3
    ai_city = AICity(args)
    ai_city.tracking()
    visualize_trajectories(join(ai_city.data_path,'AICity/train/S03/c010/vdo'), 
                           ai_city.output_path, 
                           ai_city.det_bboxes)
    idf1 = IDF1(dict_to_list_IDF1(ai_city.gt_bboxes), dict_to_list_IDF1(ai_city.det_bboxes), 0.5)
    print('IDF1: {}'.format(idf1))

if __name__ == "__main__":
    main(Config().get_args())
