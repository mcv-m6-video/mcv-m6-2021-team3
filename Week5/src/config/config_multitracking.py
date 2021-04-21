import argparse


class ConfigMultiTracking:

    def __init__(self):
        pass

    @staticmethod
    def get_args():
        parser = argparse.ArgumentParser()

        # ==================== BASE =====================
        parser.add_argument('--mode', default='color_hist', choices=['AIC2018','color_hist'])

        # =================== AIC2018 ====================
        parser.add_argument('--data_path', default='../../raw_data/AICity/train', type=str, help='Dataset path')
        parser.add_argument('--tracking_csv', default= '../outputs/tracking_iou', type=str, help='tracking result csv file')
        parser.add_argument('--output_path', default='../outputs/multitracking', help='output dir for track obj pickle & track csv')
        parser.add_argument('--dist_th', default=140, type=int, help='distance between sampled bbox centers in each track')
        parser.add_argument('--size_th', default=70, type=int, help='filter tracks that do not have detection larger than "filter size" in it longest edge')
        parser.add_argument('--mask', type=str, help='txt that describes where should be masked')
        parser.add_argument('--img_dir', default='./tmp', type=str, help='dir for saving img for reid')

        # =================== MCT =======================
        parser.add_argument('--n_layers', default=50, type=int, help='Number of layers of the network')
        parser.add_argument('--batch_size', default=10, type=int, help='Batch size')
        parser.add_argument('--dump_dir', default='./tmp', help='folder to dump images')
        parser.add_argument('--method', default='biased_knn', help='multi-camera tracking methods: cluster or bottom_up_cluster or rank')
        parser.add_argument('--cluster', default='kmeans', type=str, help='cluster methods')
        parser.add_argument('--normalize', action='store_true', help='whether normalize feature or not')
        parser.add_argument('--k', default=5, type=int, help='# of clusters')
        parser.add_argument('--n', default=4, type=int, help='bottom up parameter')
        parser.add_argument('--sum', default='avg', help='feature summarization method: max or avg')
        parser.add_argument('--filter', default=None, help='the filter file for filtering')

        return parser.parse_args()