import argparse


class Config:

    def __init__(self):
        pass

    @staticmethod
    def get_args():
        parser = argparse.ArgumentParser()

        # ============================== TRAIN PARAMS =======================================
        parser.add_argument('--data_path', type=str, default='../../data/AICity/train/S03/c010/vdo',
                            help="Path where the AICity data is located")
        parser.add_argument('--gt_path', type=str, default='../../data', help="Folder where the annotations are stored")
        parser.add_argument('--output_path', type=str, default='../outputs', help="Path to store results")
        parser.add_argument('--resize_factor', type=float, default=0.5, help="Resize factor")
        parser.add_argument('--split_factor', type=float, default=0.25, help="Split factor")
        parser.add_argument('--test_mode', type=bool, default=False, help="Test mode with less images")
        parser.add_argument('--colorspace', type=str, default="gray", help="Colorpsace to use")
        parser.add_argument('--extension', type=str, default="png", help="Extension of the frame files")
        parser.add_argument('--laplacian', type=bool, default=True, help="Use laplacian filter")
        parser.add_argument('--median_filter', type=bool, default=True, help="Use median filter")
        parser.add_argument('--bilateral_filter', type=bool, default=True, help="Use bilateral filter")
        parser.add_argument('--pre_denoise', type=bool, default=False, help="Pre denoise image")
        parser.add_argument('--alpha', type=list, default=1.71, help="Alpha for foreground estimation")
        parser.add_argument('--rho', type=float, default=0.0025, help="Rho for gaussian update")
        parser.add_argument('--noise_filter', type=list, default=['base', False], help="Type of noise filter and "
                                                                                      "whether to use connected "
                                                                                      "components")
        parser.add_argument('--fill', type=bool, default=True, help="Fill holes on the background")
        parser.add_argument('--apply_road_mask', type=bool, default=True, help="Apply ROI on background")
        parser.add_argument('--adaptive_model', type=bool, default=False, help="Use gaussian adaptive model")
        parser.add_argument('--return_bboxes', type=bool, default=True, help="Return bbox of background objects")
        parser.add_argument('--save_img', type=bool, default=True, help="Save estimated backgrounds to disk")
        parser.add_argument('--visualize', type=bool, default=True, help="Visualize bg subtraction on a window")
        parser.add_argument('--task', type=int, default=24, help="Task to do")
        parser.add_argument('--bg_model', type=str, default='base', help="Use our model (base) or state of the art")

        return parser.parse_args()
