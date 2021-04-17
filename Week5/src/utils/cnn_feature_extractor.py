import argparse
import numpy as np
from PIL import Image
import cv2

# PyTorch imports
import torch
from torch.nn import Sequential, AdaptiveAvgPool2d, CosineSimilarity,\
                     PairwiseDistance, MSELoss, L1Loss
from torch.nn.functional import adaptive_avg_pool2d
from torchvision.models import resnet50, resnet101, mobilenet_v2, vgg16
from torchvision.transforms import Normalize, Resize, Compose, ToTensor

from sklearn.metrics import jaccard_score


class CNNFeatureExtractor:
    """
        This class contains the functionality to:
            - load a pretrained model from PyTorch.
            - compute a feature vector of an image.
            - compute the distance between two feature vectors.

        Supports both GPU and CPU
    """

    def __init__(self, network='vgg', device='cpu'):
        """
            Class initializer
        """

        self.network = network  # Options: resnet50, resnet101, mobilenet, vgg
        self.device = device

        if self.network == 'resnet50':
            self.model = resnet50(pretrained=True, progress=True)
            self.model = Sequential(*list(self.model.children())[:-1])
        elif self.network == 'resnet101':
            self.model = resnet101(pretrained=True, progress=True)
            self.model = Sequential(*list(self.model.children())[:-1])
        elif self.network == 'mobilenet':
            self.model = Sequential(mobilenet_v2(pretrained=True, progress=True).features,
                                    AdaptiveAvgPool2d(output_size=(1,1)))
        elif self.network == 'vgg':
            self.model = vgg16(pretrained=True, progress=True)
            self.model = Sequential(self.model.features, 
                                    AdaptiveAvgPool2d(output_size=(1,1)))
        else:
            raise(NameError)

        if self.device in 'gpu':
            self.model.cuda()

        self.normalization = Normalize(mean=[0.485, 0.456, 0.406],
                                       std=[0.229, 0.224, 0.225])
        self.transform = Compose([ToTensor(), Resize((256, 256)), self.normalization])

    def get_image_features(self, img, bbox):
        """
            Gets the features from the image by passing it over a CNN pretrained
            on ImageNet

            :param img: 
                - either a np.array with 3 channels. The data range is 0-255.
                - or an image location (str)
            
            :return: np.array feature vector. The shape varies depending on the
                     network being used.
        """
       
        if type(img) == str:
            img = cv2.imread(img)
        
        img_h, img_w, _ = img.shape
        xmin, ymin, xmax, ymax = list(map(int, bbox))

        img = img[ymin:ymax, xmin:xmax, :]
        img = self.transform(img)

        if self.device == 'gpu':
            with torch.no_grad():
                features = self.model(img[None, ...].cuda()).cpu()
        else:
            with torch.no_grad():
                features = self.model(img[None, ...])

        features = features.numpy()
        # return features.reshape(1, features.shape[1])
        return features.reshape(features.shape[1], 1)

    def get_feature_distance(self, feat1, feat2, dist='l1'):
        """
            This function computes the distance between two feature vectors.
            Cosine and pairwise distance are available

            :param feat1: first feature array
            :param feat2: second feature array
            :return: distance between feature arrays
        """
            
        if dist == 'cosine':
            metric = CosineSimilarity()
            thr = 0.18
        elif dist == 'pairwise':
            metric = PairwiseDistance()
            thr = 0.18
        elif dist == 'mse':
            metric = MSELoss()
            thr = 0.15
        elif dist == 'l1':
            metric = L1Loss()
            thr = 0.18
        else:
            raise(NameError)

        feat1 = torch.tensor(feat1)
        feat2 = torch.tensor(feat2)
        loss = metric(feat1, feat2).numpy()

        return [np.mean(loss), thr] if dist in ['cosine', 'pairwise'] else [loss, thr]


# For testing purposes
if __name__ == '__main__':
    matcher = CNNFeatureExtractor(device='gpu')
    #img = Image.open('/home/josep/Documents/Git/mcv-m6-2021-team3/Week5/src/flor.jpeg')
    img = cv2.imread('/home/josep/Documents/Git/mcv-m6-2021-team3/Week5/src/flor2.jpeg')
    img1 = cv2.imread('/home/josep/Documents/Git/mcv-m6-2021-team3/Week5/src/flor1.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    features = matcher.get_image_features(img, [0.5, 0.5, 1, 1])
    features2 = matcher.get_image_features(img1, [0.75, 0.45, 0.4, 0.5])
    features2 = matcher.get_image_features(img1, [0.5, 0.5, 1, 1])
    dist1 = matcher.get_feature_distance(features, features2)
    dist2 = matcher.get_feature_distance(features, features2, dist='pairwise')
    dist3 = matcher.get_feature_distance(features, features2, dist='mse')
    dist4 = matcher.get_feature_distance(features, features2, dist='l1')
    print(dist1, dist2, dist3, dist4)