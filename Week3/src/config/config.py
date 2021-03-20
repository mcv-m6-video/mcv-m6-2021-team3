import argparse


class Config:

    def __init__(self):
        pass

    @staticmethod
    def get_args():
        parser = argparse.ArgumentParser()
        # ================================ FRAMEWORK ============================ #
        parser.add_argument('--framework', type=str, default='ultralytics', help='What framework to use')
        parser.add_argument('--mode', type=str, default='train', choices=['train','inference','tracking'], help='What task to perform')

        # ================================ INPUT ================================ #
        parser.add_argument('--data_path', type=str, default='../../datasets/AICity/train/S03/c010/vdo',
                            help="Path where the AICity data is located")
        parser.add_argument('--gt_path', type=str, default='../../datasets', help="Folder where the annotations are stored")
        
        parser.add_argument('--img_size', nargs='+', type=int, default=[640, 640], help='train and test image sizes')
        parser.add_argument('--split_factor', type=float, default=0.25, help="Split factor")
        parser.add_argument('--test_mode', type=bool, default=False, help="Test mode with less images")
        parser.add_argument('--extension', type=str, default="png", help="Extension of the frame files")
        
        parser.add_argument('--task', type=int, default=24, help="Task to do")
        parser.add_argument('--model', type=str, default='yolov3', choices=['faster_rcnn', 'mask_rcnn', 'retinanet', 'yolov3', 'ssd'], help="Detection model used")

        # =============================== FINETUNE =============================== #

        parser.add_argument('--conf_thres', type=float, default=0.25)
        parser.add_argument('--iou_thres', type=float, default=0.45)
        parser.add_argument('--data_yolov3', type=str, default='data/yolov3', help='data.yaml path')
        parser.add_argument('--hyp', type=str, default='data/hyp.scratch.yaml', help='hyperparameters path for finetuning')
        parser.add_argument('--epochs', type=int, default=300)
        parser.add_argument('--batch_size', type=int, default=16, help='total batch size for all GPUs')
        
        # ================================ OUTPUT ================================ #
        parser.add_argument('--output_path', type=str, default='../outputs', help="Path to store results")
        parser.add_argument('--save_json', type=bool, default=True, help="Save detection results to json")
        parser.add_argument('--view_img', type=bool, default=False, help="View detection results")
        parser.add_argument('--save_img', type=bool, default=True, help="Save detection qualitative results")
        
         # ================================ TENSORFLOW PARAMS ====================== #
        parser.add_argument('--threshold', type=float, default=0.5, help="Threshold to discard detections")

        return parser.parse_args()
