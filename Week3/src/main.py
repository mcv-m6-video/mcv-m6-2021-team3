import os
from os.path import join
from utils.ai_city import AICity
from utils.utils import write_json_file, dict_to_list_tracking
from config.config import Config
from utils.yolov3 import UltralyricsYolo

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
            test=dict_to_list_tracking(aicity.det_bboxes)

    elif args.mode in 'train':
        aicity.data_to_model()
        model = UltralyricsYolo(args=args)
        model.train()

    elif args.mode in 'eval':
        if len(aicity.det_bboxes)<1:
            aicity.inference(args.weights)
        map50, map70 = aicity.get_mAP()
        print('Evaluation of training for split {} ({}, {}): mAP50={}, mAP70={}, mIoU={}'.format(args.split[0], args.model, args.framework, map50, map70, miou))
        if args.save_img:
            aicity.visualize_task()



if __name__ == "__main__":
    main(Config().get_args())
