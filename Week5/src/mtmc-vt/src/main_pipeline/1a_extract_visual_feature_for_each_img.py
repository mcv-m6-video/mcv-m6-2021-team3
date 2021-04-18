import os
import cv2
from PIL import Image
from tqdm import tqdm
import sys
sys.path.append('/home/josep/Escriptori/mtmc-vt/src')
import json

from reid_baseline.modeling import build_model
from reid_baseline.config import cfg

import torch
from torch.backends import cudnn
from torch.nn import AdaptiveAvgPool2d
from torchvision.transforms import ToTensor, Resize, Compose, Normalize



def main():
    config_file = '/home/josep/Escriptori/mtmc-vt/src/reid_baseline/configs/track2_softmax_triple.yml'

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1

    if config_file != "":
        cfg.merge_from_file(config_file)
    cfg.freeze()

    cudnn.benchmark = True
    num_classes = 128
    model = build_model(cfg, num_classes)
    model.load_state_dict(torch.load(cfg.TEST.WEIGHT))

    if torch.cuda.is_available():
        model.cuda()

    # if torch.cuda.is_available():
    #     model.cuda()

    dataset_train_base_path = '/home/josep/Escriptori/mtmc-vt/src/aic19-track1-mtmc/train'
    dataset_test_base_path = '/home/josep/Escriptori/mtmc-vt/src/aic19-track1-mtmc/test'
    seqs_train = os.listdir(dataset_test_base_path)
    base_image_dir = '/home/josep/Escriptori/mtmc-vt/src/aic19-track1-mtmc/adjust_c_cropped_imgs'
    transform = Compose([ToTensor(), Resize((256, 256))])
    print(model)
    mean = model.bottleneck.running_mean
    var = model.bottleneck.running_var
    norm = Normalize(mean=mean,
                     std=var)
    pooling = AdaptiveAvgPool2d(output_size=1)
    print(model.bottleneck)

    for seq in seqs_train:
        for cam in os.listdir(os.path.join(dataset_test_base_path, seq)):
            print("Starting cam {}".format(cam))
            gps_file = os.path.join(dataset_test_base_path, seq, cam, 'det_gps_feature.txt')

            file_data = open(gps_file, 'r').readlines()
            features_file = open(os.path.join(dataset_test_base_path, seq, cam, 'det_reid_features.json'), 'w+')

            json_full = {
                'images': {}
            }

            for line in tqdm(file_data):
                image_name = str(line.split(',')[0])

                img = cv2.imread(os.path.join(base_image_dir, image_name))
                img = transform(img)

                if torch.cuda.is_available():
                    with torch.no_grad():
                        features = model.base(img[None, ...].cuda())
                        features = model.gap(features).cpu()

                else:
                    with torch.no_grad():
                        features = model.base(img[None, ...])
                        features = model.gap(features)

                features = features.numpy()
                features = features.reshape(features.shape[1])
                json_full['images'][image_name] = ''

                for idx, feature in enumerate(features):
                    if idx == features.shape[0] - 1:
                        json_full['images'][image_name] += str(feature) + '\n'
                    else:
                        json_full['images'][image_name] += str(feature) + ','

            json.dump(json_full, features_file)
            features_file.close()


if __name__ == '__main__':
    main()