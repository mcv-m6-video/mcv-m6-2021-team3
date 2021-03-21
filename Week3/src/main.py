import os
from os.path import join
from utils.ai_city import AICity
from utils.utils import write_json_file
from config.config import Config
from utils.yolov3 import UltralyricsYolo

def main(args):

    os.makedirs(join(args.output_path, str(args.task)), exist_ok=True)

    aicity = AICity(args)
    aicity.train_val_split()

    if args.mode in 'inference':
        if len(aicity.det_bboxes)<1:
            aicity.inference()

        print('mAP: ',aicity.get_mAP())
        
        if args.save_img:
            aicity.visualize_task()
        
        if args.tracking_mode in 'overlapping':
            aicity.compute_tracking()

    elif args.mode in 'train':
        aicity.data_to_model()
        model = UltralyricsYolo(args=args)
        model.train()

    elif args.mode in 'eval':
        if len(aicity.det_bboxes)<1:
            aicity.inference(args.weights)
        map50, map70 = aicity.get_mAP()
        print('Evaluation of training for split {} ({}, {}): mAP50={}, mAP70={}'.format(args.split[0], args.model, args.framework, map50, map70))
        if args.save_img:
            aicity.visualize_task()



if __name__ == "__main__":
    main(Config().get_args())
