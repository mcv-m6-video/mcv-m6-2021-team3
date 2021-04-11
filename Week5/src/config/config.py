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
        parser.add_argument('--tracking_mode', type=str, default='kalman', choices=['overlapping','kalman'], help='What type of tracking to perform')

        # ================================ INPUT ================================ #
        parser.add_argument('--data_path', type=str, default='/media/gemma/datasets',#'../../raw_data',
                            help="Path where the AICity data is located")
        parser.add_argument('--gt_path', type=str, default='../../datasets', help="Folder where the annotations are stored")
        
        parser.add_argument('--img_size', nargs='+', type=int, default=[640, 640], help='train and test image sizes')
        parser.add_argument('--split', nargs='+', default=['sort',1], help="Split mode and K-fold")
        parser.add_argument('--test_mode', type=bool, default=False, help="Test mode with less images")
        parser.add_argument('--extension', type=str, default="jpg", help="Extension of the frame files")
        
        parser.add_argument('--task', type=int, default=24, help="Task to do")
        parser.add_argument('--model', type=str, default='yolov3', choices=['yolov3', 'yolov3-spp', 'yolov3-tiny'], help="Detection model used")
        parser.add_argument('--weights', type=str, default=None)

        # =============================== FINETUNE =============================== #

        parser.add_argument('--seq_train', nargs='+', default=['S01','S04'], help="Sequences used to train")
        parser.add_argument('--seq_test', nargs='+', default=['S03'], help="Sequence used to test")
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
        parser.add_argument('--view_tracking', type=bool, default=True, help="Save detection qualitative results")
        
         # ================================ TENSORFLOW PARAMS ====================== #
        parser.add_argument('--threshold', type=float, default=0.5, help="Threshold to discard detections")
        # ================================ TRACKING PARAMS ====================== #
        parser.add_argument('--track_thr', type=list, default=[0.1,0.3,0.5,0.7,0.9], help="Threshold to set FP or FN")


        return parser.parse_args()
