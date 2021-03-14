import numpy as np
import cv2
from os.path import join
import glob
from tqdm import tqdm


def filter_noise(bg, post_processing=True, min_area=0.15):
    """

    """

    if post_processing:
        # Opening
        bg = cv2.morphologyEx(bg, cv2.MORPH_OPEN, np.ones((1, 1)))
        # Closing
        bg = cv2.morphologyEx(bg, cv2.MORPH_CLOSE, np.ones((3, 3)))

    # Connected components
    num_lab, labels = cv2.connectedComponents(bg)

    # Filter by area
    rm_labels = [u for u in np.unique(labels) if np.sum(labels == u) < min_area * len(bg)]
    for label in rm_labels:
        bg[np.where(labels == label)] = 0
        labels[np.where(labels == label)] = 0

    return bg, labels


def generate_bbox(mask):
    """
    
    """
    pos = np.where(mask == 255)

    y = np.min(pos[0])
    x = np.min(pos[1])
    h = np.max(pos[0]) - np.min(pos[0])
    w = np.max(pos[1]) - np.min(pos[1])

    return x, y, w, h


def fill_gaps(labels):
    """

    """
    bg_idx = np.argmax([np.sum(labels == u) for u in np.unique(labels)])
    fg_idx = np.delete(np.unique(labels), bg_idx)

    bg = np.zeros(labels.shape, dtype=np.uint8)

    for label in fg_idx:
        aux = np.zeros(labels.shape, dtype=np.uint8)
        aux[labels == label] = 255

        contours, _ = cv2.findContours(aux, 1, 2)

        # create hull array for convex hull points
        hull = []
        # calculate points for each contour
        for i in range(len(contours)):
            # creating convex hull object for each contour
            # hull = cv2.convexHull(contours[i], False)
            epsilon = 0.1 * cv2.arcLength(contours[i], True)
            approx = cv2.approxPolyDP(contours[i], epsilon, True)
            aux = cv2.fillPoly(aux, pts=[approx], color=(255))

        x, y, w, h = generate_bbox(aux)
        if np.sum(aux > 0) / (w * h) > .35:
            aux = cv2.rectangle(aux, (x, y), (x + w, y + h), (50), 2)
            bg = bg + aux

        '''drawing = np.zeros((aux.shape[0], aux.shape[1], 3), np.uint8)
        # draw contours and hull points
        for i in range(len(contours)):
            color_contours = (0, 255, 0) # green - color for contours
            color = (255, 0, 0) # blue - color for convex hull
            # draw ith contour
            cv2.drawContours(drawing, contours, i, color_contours, 1, 8, hierarchy)
            # draw ith convex hull object
            cv2.drawContours(drawing, hull, i, color, 1, 8)'''

    return bg


def get_single_objs(bg):
    """

    """
    # Filter noise
    bg, labels = filter_noise(bg)

    # Generate bboxes    
    bg = fill_gaps(labels)

    # Filter by aspect ratio
    return bg


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

            bg = get_single_objs(bg)

            '''if self.rm_noise is not None:
                bg = filter_noise(bg, self.rm_noise)
            if self.fill:
                bg = fill_gaps(bg)'''
        else:
            if self.options['colorspace'] == "LAB" or self.options['colorspace'] == "YCbCr":
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

        return bg.astype(np.uint8)

    def get_frames_background(self):
        for frame_id, frame_path in tqdm(enumerate(self.bg_frames_paths), 'Predicting background'):
            frame = self.read_frame(frame_path, colorspace=self.options['colorspace'],
                                    laplacian=self.options['laplacian'], pre_denoise=self.options['pre_denoise'])

            bg = self.get_frame_background(frame)
            img = self.read_frame(frame_path, colorspace='gray')
            img = cv2.hconcat((bg, img.astype(np.uint8)))
            # cv2.imwrite('laplacian/{}.jpg'.format(frame_id),img)
            cv2.imshow("Background", img)
            cv2.waitKey(100)

            if self.options['adaptive_model']:
                self.update_gaussian(frame, bg)

            frame, img, bg = None, None, None

    def read_frames(self):
        """
        :return:
        """

        images = []
        for file_name in tqdm(self.bg_modeling_frames_paths, 'Reading frames'):
            images.append(self.read_frame(file_name, colorspace=self.options['colorspace'],
                                          laplacian=self.options['laplacian'],
                                          pre_denoise=self.options['pre_denoise']))

        return np.asarray(images)

    def read_frame(self, path, colorspace='gray', laplacian=False, pre_denoise=False):
        """

        """

        if colorspace == 'gray':
            image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

            if self.options['resize_factor'] < 1.0:
                image = cv2.resize(image,
                                   (int(image.shape[1] * self.options['resize_factor']),
                                    int(image.shape[0] * self.options['resize_factor'])),
                                   cv2.INTER_CUBIC)
            if pre_denoise:
                image = cv2.fastNlMeansDenoising(image, templateWindowSize=7)
            if laplacian:
                image = cv2.Laplacian(image, cv2.CV_8U)

        else:
            image = cv2.imread(path)

            if self.options['resize_factor'] < 1.0:
                image = cv2.resize(image,
                                   (int(image.shape[1] * self.options['resize_factor']),
                                    int(image.shape[0] * self.options['resize_factor'])),
                                   cv2.INTER_CUBIC)
            if pre_denoise:
                image = cv2.fastNlMeansDenoising(image, templateWindowSize=7)
            if laplacian:
                image = cv2.Laplacian(image, cv2.CV_8U)

            if self.options['colorspace'] == "LAB":
                image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)[:, :, 1:]

            elif self.options['colorspace'] == "YCbCr":
                image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)[:, :, 1:]

            elif self.options['colorspace'] == "HSV":
                image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[:, :, 0]

            if self.options['median_filter']:
                filter_size = 5
                kernel = np.ones((filter_size, filter_size)) / filter_size ** 2
                image[:, :, 0] = cv2.bilateralFilter(image[:, :, 0], 9, 75, 75)
                image[:, :, 1] = cv2.bilateralFilter(image[:, :, 1], 9, 75, 75)
                # image = cv2.filter2D(image, -1, kernel)

        return image

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
