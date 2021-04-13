import argparse
import numpy as np
from PIL import Image

# PyTorch imports
import torch
from torch.nn import Sequential, AdaptiveAvgPool2d, CosineSimilarity, PairwiseDistance
from torch.nn.functional import adaptive_avg_pool2d
from torchvision.models import resnet50, resnet101, mobilenet_v2, vgg16
from torchvision.transforms import Normalize, Resize, Compose, ToTensor


class CNNFeatureExtractor:
    """
        This class contains the functionality to:
            - load a pretrained model from PyTorch.
            - compute a feature vector of an image.
            - compute the distance between two feature vectors.

        Supports both GPU and CPU
    """

    def __init__(self, network='resnet101', device='cpu'):
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

        if self.device == 'gpu':
            self.model.cuda()

        self.normalization = Normalize(mean=[0.485, 0.456, 0.406],
                                       std=[0.229, 0.224, 0.225])
        self.transform = Compose([ToTensor(), Resize((256, 256)), self.normalization])

    def get_image_features(self, img):
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
            img = Image.open(img)
        img = self.transform(img)

        if self.device == 'gpu':
            with torch.no_grad():
                features = self.model(img[None, ...].cuda()).cpu()
        else:
            with torch.no_grad():
                features = self.model(img[None, ...])

        features = features.numpy()
        return features.reshape(features.shape[1], 1)

    def get_feature_distance(self, feat1, feat2, dist='cosine'):
        """
            This function computes the distance between two feature vectors.
            Cosine and pairwise distance are available

            :param feat1: first feature array
            :param feat2: second feature array
            :return: distance between feature arrays
        """

        if dist == 'cosine':
            metric = CosineSimilarity()
        elif dist == 'pairwise':
            metric = PairwiseDistance()            
        else:
            raise(NameError)

        if self.device == 'gpu':
            feat1 = torch.tensor(feat1).cuda()
            feat2 = torch.tensor(feat2).cuda()
            return np.mean(metric(feat1, feat2).cpu().numpy())
        else:
            feat1 = torch.tensor(feat1)
            feat2 = torch.tensor(feat2)
            return np.mean(metric(feat1, feat2).numpy())


# For testing purposes
if __name__ == '__main__':
    matcher = CNNMatcher(device='gpu')
    img = Image.open('/home/josep/Documents/Git/mcv-m6-2021-team3/Week5/src/flor.jpeg')
    features = matcher.get_image_features(img)
    features2 = features * 2
    dist1 = matcher.get_feature_distance(features, features2)
    dist2 = matcher.get_feature_distance(features, features2, dist='pairwise')
    print(dist1, dist2)