import os
from os.path import join
from utils.ai_city import AICity
from utils.utils import write_json_file
from config.config import Config


def main(args):

    os.makedirs(join(args.output_path, str(args.task)), exist_ok=True)

    aicity = AICity(args)
    #aicity.inference()
    #print(aicity.get_mAP())
    aicity.train_val_split()
    aicity.visualize_task()
    #aicity.data_to_model()

    #img_path = aicity.frames_paths[0]
    #print(detect2.predict(img_path, 'faster_rcnn'))

    #tfm.predict("../../data/AICity/train/S03/c010/vdo", "SSD MobileNet V1 FPN 640x640")

if __name__ == "__main__":
    main(Config().get_args())
