import os
from os.path import join
from utils.ai_city import AICity
from utils.utils import write_json_file, dict_to_list_IDF1
from config.config import Config
from utils.yolov3 import UltralyricsYolo
from utils.visualize import visualize_tracking, visualize_trajectories
from utils.metrics import IDF1

def main(args):

    print(args.model, args.split)

    os.makedirs(join(args.output_path, str(args.task)), exist_ok=True)

    aicity = AICity(args)
    aicity.train_val_split()

    if args.mode in 'inference':
        if len(aicity.det_bboxes)<1:
            aicity.inference()

        map50, map70 = aicity.get_mAP()
        miou = aicity.get_mIoU()
        print('Inference ({}, {}): mAP50={}, mAP70={}, mIoU={}'.format(args.model, args.framework, map50, map70, miou))
        
        if args.save_img:
            aicity.visualize_task()
        
        if args.tracking_mode in 'overlapping':
            aicity.compute_tracking_overlapping()
            idf1 = IDF1 (dict_to_list_IDF1(aicity.gt_bboxes), dict_to_list_IDF1(aicity.det_bboxes))
            print('IDF1:',idf1)
            None
        elif args.tracking_mode in 'kalman':
            aicity.compute_tracking_kalman()
        #test=dict_to_list_IDF1(aicity.det_bboxes)
        if args.view_tracking:
            #visualize_tracking(aicity.data_path, aicity.output_path, aicity.det_bboxes)
            visualize_trajectories(aicity.data_path, aicity.output_path, aicity.det_bboxes)
            
    elif args.mode in 'train':
        aicity.data_to_model()
        model = UltralyricsYolo(args=args)
        model.train(args.split[1])

    elif args.mode in 'eval':
        if len(aicity.det_bboxes)<1:
            aicity.inference(args.weights)
        map50, map70 = aicity.get_mAP()
        miou = aicity.get_mIoU()
        print('Evaluation of training for split {} ({}, {}): mAP50={}, mAP70={}, mIoU={}'.format(args.split[0], args.model, args.framework, map50, map70, miou))
        if args.save_img:
            aicity.visualize_task()



if __name__ == "__main__":
    main(Config().get_args())
