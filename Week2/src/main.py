import os
from os.path import join
from utils.ai_city import AICity
from utils.utils import write_json_file
from config.config import Config


def main(args):

    os.makedirs(join(args.output_path, str(args.task)), exist_ok=True)

    alphas = [1, 1.5, 2, 3, 4, 5]
    mAP = []

    for alpha in alphas:
        args.alpha = alpha
        aicity = AICity(args)
        aicity.create_background_model()
        aicity.get_frames_background()
        mAP.append(aicity.get_mAP())
        write_json_file({'miou': aicity.miou, 'std_miou': aicity.std_iou}, str(args.alpha) + '.json')


if __name__ == "__main__":
    main(Config().get_args())
