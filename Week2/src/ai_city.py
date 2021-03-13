import numpy as np
import cv2
from os.path import join
import glob
import matplotlib.pyplot as plt
import time
from tqdm import tqdm


class AICity:
    """

    """

    def __init__(self, data_path, test_mode = False, resize_factor=0.5, denoise=False, split_factor=0.25, grayscale=True, extension="png",
                 laplacian=False, pre_denoise=False, task=1.1, alpha=3, rm_noise=False, fill=False, noise_opening=False, noise_cc=False):
        """

        """

        # PARAMETERS
        self.data_path = data_path
        self.frames_paths = glob.glob(join(self.data_path, "*." + extension))
        self.frames_paths.sort()
        self.resize_factor = resize_factor
        self.denoise = denoise
        self.split_factor = split_factor
        self.grayscale = grayscale
        self.laplacian = laplacian
        self.pre_denoise = pre_denoise
        self.task = task
        self.alpha = alpha
        self.rm_noise = rm_noise
        self.fill = fill
        self.noise_opening = noise_opening
        self.noise_cc = noise_cc
        self.test_mode = test_mode
        # FUNCTIONS
        self.split_data()

    def split_data(self):
        """

        """
        if self.test_mode:
            self.frames_paths = self.frames_paths[0:int(len(self.frames_paths)/10)]
            
        self.bg_modeling_frames_paths = self.frames_paths[
                                        :int(len(self.frames_paths) * self.split_factor)]  # 535 frames
        self.bg_frames_paths = self.frames_paths[int(len(self.frames_paths) * self.split_factor):]
        # 1606 frames

    def create_background_model(self):
        """

        """
        bg_modeling_frames = self.read_frames()
        n_frames, height, width = bg_modeling_frames.shape
        gaussian = np.zeros((height, width, 2))
        gaussian[:, :, 0] = np.mean(bg_modeling_frames, axis=0)
        gaussian[:, :, 1] = np.std(bg_modeling_frames, axis=0)

        del bg_modeling_frames

        self.background_model = gaussian

    def get_frame_background(self, frame):
        """
        :param frame:
        :param model:
        :param grayscale:
        :param rm_noise:
        :param fill:
        :return:
        """

        if self.grayscale:
            bg = np.zeros_like(frame)

            diff = frame - self.background_model[:, :, 0]
            foreground_idx = np.where(abs(diff) > self.alpha * (2 + self.background_model[:, :, 1]))

            bg[foreground_idx[0], foreground_idx[1]] = 255

            if self.rm_noise:
                bg = self.filter_noise(bg)
            if self.fill:
                bg = self.fill_gaps(bg)
        else:
            pass

        return bg

    def get_frames_background(self):
        for frame_path in self.bg_frames_paths:
            frame = self.read_frame(frame_path, laplacian=self.laplacian, pre_denoise=self.pre_denoise)
            bg = self.get_frame_background(frame)
            img = self.read_frame(frame_path)
            img = cv2.hconcat((bg, img))
            cv2.imshow("Background", img)
            cv2.waitKey(100)

    def read_frames(self):
        """
        :param paths:
        :param grayscale:
        :return:
        """

        images = []
        for file_name in tqdm(self.bg_modeling_frames_paths, 'Reading frames'):
            images.append(self.read_frame(file_name, laplacian=self.laplacian, pre_denoise=self.pre_denoise))
        
        return np.asarray(images)

    def read_frame(self, path, laplacian=False, pre_denoise=False):
        """

        """

        if self.grayscale:
            image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if self.resize_factor < 1.0:
                image = cv2.resize(image,
                                   (int(image.shape[1] * self.resize_factor), int(image.shape[0] * self.resize_factor)),
                                   cv2.INTER_CUBIC)
            if pre_denoise:   
                image = cv2.fastNlMeansDenoising(image, templateWindowSize = 5)
            if laplacian:
                image = cv2.Laplacian(image, cv2.CV_8U)
            return image
        else:
            # TODO
            pass

    def filter_noise(self, bg):
        """

        """
        if self.noise_opening == True:
            bg = cv2.morphologyEx(bg, cv2.MORPH_OPEN, (5,5))

        elif self.noise_cc == True:
            num_lab, labels = cv2.connectedComponents(bg)
            rm_labels = [u for u in np.unique(labels) if np.sum(labels == u) < 10]
            for label in rm_labels:
                bg[np.where(bg == label)] = 0

        return bg

    def fill_gaps(self, bg):
        """

        """
        return bg
