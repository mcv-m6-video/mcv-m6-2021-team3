import argparse


class ConfigMultiTracking:

    def __init__(self):
        pass

    @staticmethod
    def get_args():
        parser = argparse.ArgumentParser()

        # =================== AIC2018 ====================
        parser.add_argument('--tracking_csv', default= '../outputs/tracking_iou', type=str, help='tracking result csv file')
        parser.add_argument('--output_path', default='../outputs/multitracking', help='output dir for track obj pickle & track csv')
        parser.add_argument('--dist_th', default=140, type=int, help='distance between sampled bbox centers in each track')
        parser.add_argument('--size_th', default=70, type=int, help='filter tracks that do not have detection larger than "filter size" in it longest edge')
        parser.add_argument('--mask', type=str, help='txt that describes where should be masked')
        parser.add_argument('--img_dir', default='./tmp', type=str, help='dir for saving img for reid')

        return parser.parse_args()