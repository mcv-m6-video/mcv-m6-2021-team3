import os
import cv2
import matplotlib.pyplot as plt
from metrics import compute_iou
from utils import dict_to_list
import numpy as np
import flow_vis
from os.path import join
from tqdm import tqdm


def plot_metrics_OF(seq, gt_of, det_of, dif):
    plt.figure()
    plt.title(seq)

    plt.subplot(2, 2, 1)
    plt.imshow(gt_of[seq])
    plt.title('Ground Truth Optical Flow')

    plt.subplot(2, 2, 3)
    plt.imshow(det_of[seq])
    plt.title('Estimated Optical Flow')

    plt.subplot(2, 2, 2)
    plt.imshow(dif[seq][1])
    plt.title('Optical Flow Error')
    plt.colorbar()

    plt.subplot(2, 2, 4)
    plt.hist(dif[seq][0], bins=100)
    plt.xlabel('Optical Flow Error')
    plt.ylabel('Num of pixels')
    plt.title('Optical Flow Histogram Error')

    plt.show()


def OF_quiver_visualize(img, flow, step, fname_output='flow_quiver.png'):
    u = flow[:, :, 0]
    v = flow[:, :, 1]
    occ = flow[:, :, 2]

    U = u * occ
    V = v * occ

    (h, w) = flow.shape[0:2]

    M = np.hypot(u, v)  # color

    x, y = np.meshgrid(np.arange(0, w), np.arange(0, h))  # initial

    plt.figure()
    plt.imshow(img, cmap='gray')
    plt.quiver(x[::step, ::step], y[::step, ::step], U[::step, ::step], V[::step, ::step],
               scale_units='xy', angles='xy', scale=.05, color=(1,0,0,1))
    plt.axis('off')
    plt.savefig(fname_output)


def OF_hsv_visualize(flow, fname_output='flow_hsv.png', enhance=False):
    occ = flow[:, :, 2]
    u = flow[:, :, 0] * occ
    v = flow[:, :, 1] * occ
    magnitude, angles = cv2.cartToPolar(u.astype(np.float32), v.astype(np.float32))

    hsv = np.zeros_like(flow)
    hsv[:, :, 0] = angles * 180 / np.pi / 2
    hsv[:, :, 1] = 255
    hsv[:, :, 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

    if enhance:
        hsv[:, :, 2] *= 5

    rgb = cv2.cvtColor(hsv.astype(np.float32), cv2.COLOR_HSV2RGB)

    cv2.imwrite(fname_output, rgb)


def OF_colorwheel_visualize(flow, fname_output='flow_colorwheel.png', enhance=False):
    flow_color = flow_vis.flow_to_color(flow[:, :, :2], convert_to_bgr=False)

    # To improve the visualization
    if enhance:
        flow_hsv = cv2.cvtColor(flow_color, cv2.COLOR_RGB2HSV)
        flow_hsv[:,:,1] = flow_hsv[:,:,1]*2
        flow_hsv[:,:,2] = flow_hsv[:,:,2]*5
        flow_color = cv2.cvtColor(flow_hsv, cv2.COLOR_HSV2RGB)

    plt.figure()
    plt.imshow(flow_color)
    plt.axis('off')
    plt.savefig(fname_output)


'''def visual_of(im, gtx, gty, gtz, fname, overlap=0.9, wsize=300, mult=1, thickness=1):
    def getCoords(i, j, w_size, h, w):
        if i < 0:
            ai = 0
        else:
            ai = i

        if j < 0:
            aj = 0
        else:
            aj = j

        if i + w_size >= h:
            bi = h - 1
        else:
            bi = i + w_size

        if j + w_size >= h:
            bj = w - 1
        else:
            bj = j + w_size

        return int(ai), int(bi), int(aj), int(bj)

    step = int(wsize * (1 - overlap))
    mwsize = int(wsize / 2)
    h,w = gtx.shape

    for i in tqdm(np.arange(-mwsize,h+1-mwsize,step)):
        for j in tqdm(np.arange(-mwsize,w+1-mwsize,step)):
            ai,bi, aj, bj = getCoords(i, j, wsize, h, w)
            mask = gtz[ai:bi, aj:bj]
            mask = mask.astype(np.int8)
            if np.count_nonzero(mask) == 0:
                continue
            winx = gtx[ai:bi, aj:bj]
            winy = gty[ai:bi, aj:bj]
            glob_x = (np.sum(winx[mask])*mwsize)/(np.count_nonzero(mask)*512)*mult
            glob_y = (np.sum(winy[mask])*mwsize)/(np.count_nonzero(mask)*512)*mult
            pt1 = (int(j + mwsize), int(i + wsize / 2))
            pt2 = (int(j + mwsize + glob_x), int(i + mwsize + glob_y))
            color = (0, 255, 0)
            im = cv2.arrowedLine(im, pt1, pt2, color, thickness)

    cv2.imwrite(fname, im)'''


def draw_bboxes(img, bboxes, color):
    for bbox in bboxes:
        bbox = bbox.astype(int)
        img = cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
    return img


def visualize_iou(gt, dets, frames, det_model):
    miou = np.empty(0, )
    os.makedirs(join('./task_2', det_model), exist_ok=True)

    for frame in frames:
        if os.name == 'nt':
            frame = frame.replace(os.sep, '/')
        frame_id = (frame.split('/')[-1]).split('.')[0]

        if frame_id in gt.keys() and int(frame_id) > 210:
            gt_frame = np.array(dict_to_list(gt[frame_id], False))
            dets_frame = np.array(dict_to_list(dets[frame_id], False))

            miou = np.hstack((miou, compute_miou(gt_frame, dets_frame, frame_id)))

            plt.figure(figsize=(10, 12))

            plt.subplot(2, 1, 1)
            img = cv2.cvtColor(cv2.imread(frame), cv2.COLOR_BGR2RGB)
            img = draw_bboxes(img, gt_frame, (0, 255, 0))
            img = draw_bboxes(img, dets_frame, (0, 0, 255))
            plt.imshow(img)

            plt.subplot(2, 1, 2)
            plt.plot(np.arange(210, 210 + len(miou), 1), miou, 'b')
            plt.axis([210, 1210, 0, 1])
            plt.xlabel('Frame id')
            plt.ylabel('mIoU')

            plt.savefig(join('./task_2', det_model, frame_id + '.png'))
            plt.close()