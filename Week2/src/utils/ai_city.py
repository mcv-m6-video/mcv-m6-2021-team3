import numpy as np
import cv2
from os.path import join
import glob
from tqdm import tqdm
from utils.refinement import get_single_objs, filter_noise

class AICity:
    """

    """

    def __init__(self, data_path, options):
        """

        """

        # PARAMETERS
        self.data_path = data_path
        self.frames_paths = glob.glob(join(self.data_path, "*." + options['extension']))
        self.frames_paths.sort()
        self.options = options

        # FUNCTIONS
        self.split_data()

    def split_data(self):
        """

        """
        if self.options['test_mode']:
            self.frames_paths = self.frames_paths[0:int(len(self.frames_paths) / 10)]

        self.bg_modeling_frames_paths = self.frames_paths[
                                        :int(len(self.frames_paths) * self.options['split_factor'])]  # 535 frames
        self.bg_frames_paths = self.frames_paths[int(len(self.frames_paths) * self.options['split_factor']):]
        # 1606 frames

    def create_background_model(self):
        """

        """
        bg_modeling_frames = self.read_frames()

        if self.options['colorspace'] == 'gray':
            n_frames, height, width = bg_modeling_frames.shape
            gaussian = np.zeros((height, width, 2))
            gaussian[:, :, 0] = np.mean(bg_modeling_frames, axis=0)
            gaussian[:, :, 1] = np.std(bg_modeling_frames, axis=0)

            del bg_modeling_frames

            self.background_model = gaussian

        else:
            if self.options['colorspace'] == "LAB":
                n_frames, height, width, n_channels = bg_modeling_frames.shape
                gaussian = np.zeros((height, width, 4))

                # Channel A
                gaussian[:, :, 0] = np.mean(bg_modeling_frames[:, :, :, 0], axis=0)
                gaussian[:, :, 1] = np.std(bg_modeling_frames[:, :, :, 0], axis=0)

                # Channel B
                gaussian[:, :, 2] = np.mean(bg_modeling_frames[:, :, :, 1], axis=0)
                gaussian[:, :, 3] = np.std(bg_modeling_frames[:, :, :, 1], axis=0)

                self.background_model = gaussian

            elif self.options['colorspace'] == "HSV":
                n_frames, height, width = bg_modeling_frames.shape

                gaussian = np.zeros((height, width, 2))

                # Channel H
                gaussian[:, :, 0] = np.mean(bg_modeling_frames, axis=0)
                gaussian[:, :, 1] = np.std(bg_modeling_frames, axis=0)

                self.background_model = gaussian

    def get_frame_background(self, frame):
        """
        :param frame:
        :return:
        """

        if self.options['colorspace'] == 'gray':
            bg = np.zeros_like(frame)

            diff = frame - self.background_model[:, :, 0]
            foreground_idx = np.where(abs(diff) > self.options['alpha'] * (2 + self.background_model[:, :, 1]))

            bg[foreground_idx[0], foreground_idx[1]] = 255

        elif self.options['colorspace'] == "LAB" or self.options['colorspace'] == "YCbCr":
            bg = np.zeros((frame.shape[0], frame.shape[1]))

            diff_ch1 = frame[:, :, 0] - self.background_model[:, :, 0]
            diff_ch2 = frame[:, :, 1] - self.background_model[:, :, 2]

            foreground_ch1_idx = np.where(abs(diff_ch1) > self.options['alpha'] * (2 + self.background_model[:, :, 1]))
            foreground_ch2_idx = np.where(abs(diff_ch2) > self.options['alpha'] * (2 + self.background_model[:, :, 3]))

            bg[foreground_ch1_idx[0], foreground_ch1_idx[1]] = 255
            bg[foreground_ch2_idx[0], foreground_ch2_idx[1]] = 255

        elif self.options['colorspace'] == "HSV":
            bg = np.zeros_like(frame)

            diff = frame - self.background_model[:, :, 0]

            foreground_idx = np.where(abs(diff) > self.options['alpha'] * (2 + self.background_model[:, :, 1]))

            bg[foreground_idx[0], foreground_idx[1]] = 255

        if self.options['return_bboxes']:
            bg, bboxes = get_single_objs(bg, self.options['noise_filter'], self.options['fill'])
            return bg, bboxes

        elif not self.options['return_bboxes'] and self.options['noise_filter']:
            # Filter noise
            bg, _ = filter_noise(bg, self.options['noise_filter'])
        
        return bg.astype(np.uint8), None

    def get_frames_background(self):
        for frame_id, frame_path in tqdm(enumerate(self.bg_frames_paths[150:]), 'Predicting background'):
            img, frame = self.read_frame(frame_path, colorspace=self.options['colorspace'],
                                    laplacian=self.options['laplacian'], pre_denoise=self.options['pre_denoise'])

            bg, bboxes = self.get_frame_background(frame)
            bg = np.repeat(np.expand_dims(bg,axis=2),3,axis=2)

            if self.options['adaptive_model']:
                self.update_gaussian(frame, bg)
            
            if self.options['return_bboxes']:
                for x,y,w,h in bboxes:
                    bg = cv2.rectangle(bg,(x,y),(x+w,y+h),(255,0,0),2)
                    img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

            img = cv2.hconcat((bg, img))
            cv2.imwrite('result/{}.jpg'.format(frame_id),img)
            #cv2.imshow("Background", img)
            #cv2.waitKey(100)
            
            # Free memory
            del img, frame, bg

    def read_frames(self):
        """
        :return:
        """

        images = []
        for file_name in tqdm(self.bg_modeling_frames_paths, 'Reading frames'):
            images.append(self.read_frame(file_name, colorspace=self.options['colorspace'],
                                          laplacian=self.options['laplacian'],
                                          pre_denoise=self.options['pre_denoise'])[1])

        return np.asarray(images)

    def read_frame(self, path, colorspace='gray', laplacian=False, pre_denoise=False):
        """

        """
        img0 = cv2.imread(path)

        if self.options['resize_factor'] < 1.0:
            img = cv2.resize(img0,
                                (int(img0.shape[1] * self.options['resize_factor']),
                                int(img0.shape[0] * self.options['resize_factor'])),
                                cv2.INTER_CUBIC)
        if pre_denoise:
            img = cv2.fastNlMeansDenoising(img, templateWindowSize=7)
        if laplacian:
            img = cv2.Laplacian(img, cv2.CV_8U)

        if self.options['colorspace'] == 'gray':
            img = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)

        elif self.options['colorspace'] == "LAB":
            img = cv2.cvtColor(img0, cv2.COLOR_BGR2LAB)[:, :, 1:]

        elif self.options['colorspace'] == "YCbCr":
            img = cv2.cvtColor(img0, cv2.COLOR_BGR2YCrCb)[:, :, 1:]

        elif self.options['colorspace'] == "HSV":
            img = cv2.cvtColor(img0, cv2.COLOR_BGR2HSV)[:, :, 0]

        if self.options['bilateral_filter']:            
            img[:, :, 0] = cv2.bilateralFilter(img[:, :, 0], 9, 75, 75)
            img[:, :, 1] = cv2.bilateralFilter(img[:, :, 1], 9, 75, 75)
        
        if self.options['median_filter']:
            filter_size = 5
            kernel = np.ones((filter_size, filter_size)) / filter_size ** 2
            img = cv2.filter2D(img, -1, kernel)

        return img0, img

    def update_gaussian(self, frame, bg):
        """

        """

        if self.options['colorspace'] == 'gray':
            [x, y] = np.where(bg == 0)
            # update mean
            self.background_model[x, y, 0] = self.options['rho'] * frame[x, y] + (1 - self.options['rho']) * self.background_model[x, y, 0]
            # update std
            self.background_model[x, y, 1] = self.options['rho'] * np.square(frame[x, y] - self.background_model[x, y, 0]) + (
                        1 - self.options['rho']) * self.background_model[x, y, 1]
        else:
            pass
