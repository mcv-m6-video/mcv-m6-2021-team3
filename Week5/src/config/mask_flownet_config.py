import argparse


class MaskFlownetConfig:

    def __init__(self):
        pass

    @staticmethod
    def get_args():
        parser = argparse.ArgumentParser()
        
        # MaskFlownet
        parser.add_argument('--config', type=str, nargs='?', default='MaskFlownet.yaml')
        parser.add_argument('--video_filepath', type=str, help='filepath of the input video')
        parser.add_argument('--gpu_device', type=str, default='0', help='Specify gpu device(s)')
        parser.add_argument('--checkpoint', type=str, default='8caNov12', help='model checkpoint to load; by default, the latest one.'
                            'You can use checkpoint:steps to load to a specific steps')
        parser.add_argument('--clear_steps', action='store_true')
        parser.add_argument('-n', '--network', type=str, default='MaskFlownet', help='The choice of network')
        parser.add_argument('--batch', type=int, default=8, help='minibatch size of samples per device')
        parser.add_argument('--threads', type=str, default=8, help='Number of threads to use when writing flow video to file')

        return parser.parse_args()
