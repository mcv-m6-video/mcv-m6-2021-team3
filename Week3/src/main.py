import os
from os.path import join
from utils.ai_city import AICity
from utils.utils import write_json_file, dict_to_list_IDF1
from config.config import Config
from utils.yolov3 import UltralyricsYolo
from utils.visualize import visualize_trajectories, plot_idf1_thr
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
        
        if args.tracking_mode in ['overlapping','kalman']:
            if args.tracking_mode in 'overlapping':
                aicity.compute_tracking_overlapping()
            elif args.tracking_mode in 'kalman':
                aicity.compute_tracking_kalman()
            thresholds = args.track_thr
            idf1_thr = []
            for thr in thresholds:
                idf1 = IDF1 (dict_to_list_IDF1(aicity.gt_bboxes), dict_to_list_IDF1(aicity.det_bboxes), thr)
                idf1_thr.append(idf1)
                print('IDF1:',idf1)
            plot_idf1_thr(aicity.output_path, idf1_thr, thresholds)
        if args.view_tracking:
            visualize_trajectories(aicity.data_path, aicity.output_path, aicity.det_bboxes)
            
    elif args.mode in 'train':
        aicity.data_to_model()
        model = UltralyricsYolo(args=args)
        if args.split[1]>1:
            model.train(args.k)
        else:
            model.train()

    elif args.mode in 'eval':
        if len(aicity.det_bboxes)<1:
            aicity.inference(args.weights)
        map50, map70 = aicity.get_mAP(args.k)
        miou = aicity.get_mIoU()
        print('Evaluation of training for split {} ({}, {}): mAP50={}, mAP70={}, mIoU={} | conf_thres={}, iou_thres={}'.format(args.split[0], args.model, args.framework, map50, map70, miou,
                args.conf_thres, args.iou_thres))

        if args.save_img:
            aicity.visualize_task()

if __name__ == "__main__":
    main(Config().get_args())