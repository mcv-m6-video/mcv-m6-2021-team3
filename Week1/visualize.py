import os
import cv2
import numpy as np
from os.path import join
import flow_vis
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
from metrics import compute_miou
from utils import dict_to_list
import imageio


def plot_metrics_OF(seq, gt_of, det_of, dif):
    """
    Plot the Optical Flow from the GT, the Estimated, the 
    difference between both and a histogram of the error
    :param seq: name of the sequence used
    :param gt_of: Optical Flow GT 
    :param det_of: Optical Flow Estimated
    :param dif: difference between the GT and the estimated OF
    """
    fig = plt.figure(figsize=(16,8))
    plt.subplot(2,2,1)
    plt.imshow(gt_of[seq])
    plt.title('Ground Truth Optical Flow')

    plt.subplot(2,2,3)
    plt.imshow(det_of[seq])
    plt.title('Estimated Optical Flow')

    plt.subplot(2,2,2)
    ax = plt.gca()
    im = ax.imshow(dif[seq][1])
    plt.title('Optical Flow Error')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)

    plt.subplot(2,2,4)
    plt.hist(dif[seq][0],bins=100, color='cadetblue')
    plt.xlabel('Optical Flow Error')
    plt.ylabel('Num of pixels')
    plt.title('Optical Flow Histogram Error')

    plt.savefig(seq+'.png')


def OF_quiver_visualize(img, flow, step, fname_output='flow_quiver.png'):
    """
    Plot the OF through quiver function
    :param img: the scene RGB image
    :param flow: the Optical flow image (GT or Estimated)
    :param step: Step controls the sampling to draw the arrows 
    :param fname_output: name given to the output image to be saved
    """
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
               M[::step, ::step], scale_units='xy', angles='xy', scale=.05, color=(1,0,0,1))
    plt.axis('off')
    plt.savefig(fname_output)


def OF_hsv_visualize(flow, fname_output='flow_hsv.png', enhance=False):
    """
    Plot the Optical Flow using HSV color space
    :param flow: Optical Flow image (GT or Estaimation)
    :param fname_output: name given to the output image to be saved
    :param enhance: A boolean parameter to control if the visualization is improved or not
    """
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
    """
    Plot the Optical Flow using Wheel color
    :param flow: Optical Flow image (GT or Estaimation)
    :param fname_output: name given to the output image to be saved
    :param enhance: A boolean parameter to control if the visualization is improved or not

    https://github.com/tomrunia/OpticalFlow_Visualization
    S. Baker, D. Scharstein, J. Lewis, S. Roth, M. J. Black, and R. Szeliski.
    A database and evaluation methodology for optical flow.
    In Proc. IEEE International Conference on Computer Vision (ICCV), 2007.
    """
    flow_color = flow_vis.flow_to_color(flow[:, :, :2], convert_to_bgr=False)

    # To improve the visualization
    if enhance:
        flow_hsv = cv2.cvtColor(flow_color, cv2.COLOR_RGB2HSV)
        flow_hsv[:,:,1] = flow_hsv[:,:,1]*2
        flow_hsv[:,:,2] = flow_hsv[:,:,2]*5
        flow_color = cv2.cvtColor(flow_hsv, cv2.COLOR_HSV2RGB)

    cv2.imwrite(fname_output,flow_color)


def draw_bboxes(img, bboxes, color):
    """
    Draw bounding boxes onto an image
    :param img: Input RGB image of the scene
    :param bboxes: list of coordinates of the bounding boxes to be drawn
    :param color: color used to draw the bounding boxes
    :return: image with the bounding boxes drawn
    """

    for bbox in bboxes:
        bbox = bbox.astype(int)
        img = cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
    return img


def visualize_iou(gt, dets, frames, det_model, save_dir='./task2'):
    """
    Plot the graphic of the IOU metric
    :param gt: list with the bounding boxes of the GT file
    :param dets: list with the detected bounding boxes
    :param frames: frames extracted from the video
    :param det_model: model used (Mask RCNN, SSD512, YOLO3)
    :param save_dir: path to where gif will be saved
    """
    os.makedirs(join(save_dir,det_model),exist_ok=True)

    gif_dir = join(save_dir,det_model+'.gif')

    if os.path.exists(gif_dir):
        print('Gif saved at '+gif_dir)
        return


    miou, std_iou = np.empty(0,), np.empty(0,)

    with imageio.get_writer(gif_dir, mode='I') as writer:

        for frame in tqdm(frames[499:800],'Evaluating detections from {} at each frame'.format(det_model)):
            if os.name == 'nt':
                frame = frame.replace(os.sep, '/')
            frame_id = (frame.split('/')[-1]).split('.')[0]

            if frame_id in gt.keys():
                gt_frame = np.array(dict_to_list(gt[frame_id],False))
                dets_frame = np.array(dict_to_list(dets[frame_id],False))
                
                mean, std = compute_miou(gt_frame,dets_frame,frame_id)
                miou = np.hstack((miou,mean))
                std_iou = np.hstack((std_iou,std))

                plt.figure(figsize=(5,6))

                plt.subplot(2,1,1)
                img = cv2.cvtColor(cv2.imread(frame),cv2.COLOR_BGR2RGB)
                img = draw_bboxes(img,gt_frame,(0,255,0))
                img = draw_bboxes(img,dets_frame,(0,0,255))
                plt.imshow(img)
                plt.plot(0, 0, "-", c=(0,1,0), label='Ground Truth')
                plt.plot(0, 0, "-", c=(0,0,1), label='Detection')
                plt.legend(prop={'size': 8},loc='lower right')
                
                xaxis = np.arange(500,500+len(miou),1)
                
                plt.subplot(2,1,2)
                plt.plot(xaxis,miou,'cadetblue',label='Mean IoU')
                plt.fill(np.append(xaxis, xaxis[::-1]), np.append(miou+std_iou,(miou-std_iou)[::-1]), 'powderblue', label='STD IoU')
                plt.axis([500,800,0,1])
                plt.xlabel('Frame id',fontsize=10)
                plt.ylabel('IoU',fontsize=10)
                plt.legend(prop={'size': 8},loc='lower right')

                plt.savefig(join(save_dir,det_model,frame_id+'.png'))
                plt.close()

                image = imageio.imread(join(save_dir,det_model,frame_id+'.png'))
                writer.append_data(image)
    
    print('Gif saved at '+gif_dir)