import os
from os.path import join
from utils.ai_city import AICity
from utils.utils import write_json_file
from config.config import Config
from utils.yolov3 import UltralyricsYolo

def main(args):

    os.makedirs(join(args.output_path, str(args.task)), exist_ok=True)

    if args.mode in 'inference':

        models = ['SSD_MobileNet_V1_FPN_640x640',
                    'SSD_MobileNet_V2_FPNLite_640x640',
                    'SSD_ResNet101_V1_FPN_640x640_(RetinaNet101)',
                    'SSD_ResNet152_V1_FPN_640x640_(RetinaNet152)',
                    'EfficientDet_D1_640x640',
                    'Faster_R-CNN_ResNet101_V1_1024x1024',
                    'Faster_R-CNN_ResNet101_V1_640x640',
                    'CenterNet_Resnet101_V1_FPN_512x512',
                    'Mask_R-CNN_Inception_ResNet_V2_1024x1024']

        for model in models:
            args.model = model
            aicity = AICity(args)
            aicity.train_val_split()

            if len(aicity.det_bboxes)<1:
                aicity.inference()

            print(model)
            print('mAP 50: ',aicity.get_mAP(map_70=False))
            print('mAP 70: ',aicity.get_mAP(map_70=True))
            print('mIoU: ', aicity.get_mIoU())
        
            if args.save_img:
                aicity.visualize_task()

    elif args.mode in 'train':
        if args.framework == 'ultralytics':
            aicity.data_to_model()
            model = UltralyricsYolo(args=args)
            model.train()
        elif args.framework == 'tensorflow':
            aicity.data_to_model()
            #aicity.train()

    else:
        raise NameError
        


if __name__ == "__main__":
    main(Config().get_args())
