import argparse


class Config:

    def __init__(self):
        pass

    @staticmethod
    def get_args():
        parser = argparse.ArgumentParser()
        # ================================ FRAMEWORK ============================ #

        parser.add_argument('--framework', type=str, default='ultralytics', help='What framework to use')
        parser.add_argument('--mode', type=str, default='inference', choices=['train','eval','inference','tracking'], help='What task to perform')
        parser.add_argument('--tracking_mode', type=str, default='overlapping', choices=['overlapping','kalman'], help='What type of tracking to perform')

        # ================================ INPUT ================================ #
        parser.add_argument('--data_path', type=str, default='../../raw_data/AICity/train/S03/c010/vdo',
                            help="Path where the AICity data is located")
        parser.add_argument('--gt_path', type=str, default='../../raw_data', help="Folder where the annotations are stored")
        parser.add_argument('--img_size', nargs='+', type=int, default=[640, 640], help='train and test image sizes')
        parser.add_argument('--split', nargs='+', default=['sort',1], help="Split mode and K-fold")
        parser.add_argument('--k', type=int, default=None, help="k")
        parser.add_argument('--test_mode', type=bool, default=False, help="Test mode with less images")
        parser.add_argument('--extension', type=str, default="png", help="Extension of the frame files")
        parser.add_argument('--task', type=int, default=24, help="Task to do")
        parser.add_argument('--model', type=str, default='yolov3', choices=['faster_rcnn', 'mask_rcnn', 'retinanet', 'yolov3', 'yolov3-spp',
                                                                            'yolov3-tiny','mobilenet64', 'resnet640', 'efficientdetd1'], help="Detection model used")
        parser.add_argument('--weights', type=str, default='runs/train/yolov3_sort/weights/best.pt')

        # =============================== FINETUNE =============================== #
        parser.add_argument('--conf_thres', type=float, default=0.2)
        parser.add_argument('--iou_thres', type=float, default=0.3)
        parser.add_argument('--data_yolov3', type=str, default='data/finetune/yolov3/cars_rand.yaml', help='data.yaml path')
        parser.add_argument('--hyp', type=str, default='data/finetune/yolov3/hyp.finetune.yaml', help='hyperparameters path for finetuning')
        parser.add_argument('--epochs', type=int, default=150)
        parser.add_argument('--batch_size', type=int, default=32, help='total batch size for all GPUs')

        # ================================ OUTPUT ================================ #
        parser.add_argument('--output_path', type=str, default='../outputs', help="Path to store results")
        parser.add_argument('--save_json', type=bool, default=True, help="Save detection results to json")
        parser.add_argument('--view_img', type=bool, default=False, help="View detection results")
        parser.add_argument('--save_img', type=bool, default=False, help="Save detection qualitative results")
        parser.add_argument('--view_tracking', type=bool, default=False, help="Save detection qualitative results")
        
         # ================================ TENSORFLOW PARAMS ====================== #
        parser.add_argument('--iou_threshold', type=float, default=0.5, help="Threshold to discard detections")
        parser.add_argument('--tf_records_path', type=str, default='./data/finetune/tf_records', help='Path to store tfrecords')
        parser.add_argument('--model_conf_file', type=str, default='ssd_resnet152_v1_fpn_1024x1024_coco17_tpu-8.config')
        parser.add_argument('--coco_model', type=bool, default=False, help="Wether the model is trained on COCO or AICity")
        parser.add_argument('--trained_model', type=str, default="resnet640", help="Folder containing the trained model")

        # ================================ TRACKING PARAMS ====================== #
        parser.add_argument('--track_thr', type=list, default=[0.1,0.3,0.5,0.7,0.9], help="Threshold to set FP or FN")

        return parser.parse_args()
