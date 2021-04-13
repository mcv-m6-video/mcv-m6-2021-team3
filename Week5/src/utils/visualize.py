import os
import cv2
import glob
import numpy as np
from os.path import join
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from utils.metrics import compute_miou, compute_centroid
from utils.utils import dict_to_list, read_json_file, str_frame_id
from itertools import compress
from matplotlib import cm
from matplotlib.ticker import MaxNLocator
from matplotlib.animation import FuncAnimation
import flow_vis

def plot_idf1_thr(path_out ,idf1, thrs):
    plt.plot(thrs, idf1, color='paleturquoise')
    plt.xlabel('Thresholds')
    plt.ylabel('IDF1')
    plt.savefig(join(path_out,'idf1')+'.png')

def visualize_trajectories(path_in, path_out, det_bboxes):
    """
    Computes the trajectories bboxes and center movement along the frames. 
    Each object keeps an UI a color. 
    :param path_in: path where the frames are grab
    :param path_in: path where the frames are saved
    :param det_bboxes: dictionary with the information of the detections
    """
    # not assuming any order
    start_frame = int(min(det_bboxes.keys()))
    num_frames = int(max(det_bboxes.keys())) - start_frame + 1

    id_ocurrence = {}
    # Count ocurrences and compute centers 
    for i in range(start_frame, num_frames):
        if str_frame_id(i) not in det_bboxes.keys():
            continue
        for detection in det_bboxes[str_frame_id(i)]:
            # Store story of obj_id along with their centroids
            objt_id = detection['obj_id']
            if objt_id in id_ocurrence:
                id_ocurrence[objt_id].append((i,compute_centroid(detection['bbox'])))
            else:
                id_ocurrence[objt_id] = [(i,compute_centroid(detection['bbox']))] 
    # Ensure unique color for ID
    num_colors = 1000
    colours = np.random.rand(num_colors,3) 
    for i in tqdm(range(start_frame, num_frames),"saving tracking img"):
        f_id = str_frame_id(i)
        detections = []
        id_list = []
        
        if str_frame_id(i) in det_bboxes.keys():
            frame = det_bboxes[f_id]
            for detection in frame:
                bb_id = detection['obj_id']
                bbbox = detection['bbox']
                detections.append(bbbox)
                id_list.append(bb_id)

        img = draw_frame_track(path_in, f_id, detections, colours, id_list, id_ocurrence) 
        #cv2.imshow('Tracking',img)
        #cv2.waitKey(1)
        os.makedirs(join(path_out, 'tracking'), exist_ok=True)
        cv2.imwrite(join(path_out,'tracking',f_id)+'.jpg',img)
        

def draw_frame_track(path_in, frame, detections, colors, ids, id_ocurrence=[]):
    """
    :param path_in: path where the frames are saved
    :param frame: frame id
    :param detections: the different detections achieved in each frame
    :param colors: colors needed to print the diferent bboxes
    :param fill: a boolean to decide if the bbox is filled or not
    :return: return the image created by the frame and its bboxes
    """
    img = cv2.imread(join(path_in,frame)+'.jpg')
    for detection, bb_id in zip(detections, ids):
        #get color and draw bbox with id
        color = colors[bb_id%1000,:]*255
        img = cv2.rectangle(img, (int(detection[0]),int(detection[1])), (int(detection[2]),int(detection[3])), tuple(color), 3)
        img = cv2.putText(img, str(bb_id), (int(detection[0]),int(detection[1])-10),cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        #draw trajectories while the id is on the frmame
        if id_ocurrence:
            for track_id, tracking  in id_ocurrence.items():
                c_start = 0
                color = colors[track_id%1000,:]*255
                if (c_start == 0) and (tracking[-1][0] < int(frame)):
                    continue
                for f_id, c_end in tracking:
                    if f_id < int(frame):
                        if c_start:
                            img = cv2.line(img, c_start, c_end, color, 2)
                        c_start = c_end
    return img

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


def visualize_background_iou(data, segmen, gt, dets, framework, model, output_path, mode='inference', axis=[536, 915]):
    """
    Creates a plot to visualize the IoU with its mean and std deviation.

    :param data: path to train and val images
    :param segmen: segmentation estimated
    :param gt: ground truth
    :param dets: bbox detected
    :param opt: options/config used in the background removal process
    :param axis: list of the frame ids evaluated
    """
    miou, std_iou, xaxis = np.empty(0, ), np.empty(0, ), np.empty(0, )
    frames_paths = data['train']+data['val']
    frames_paths.sort()
    for file_name in tqdm(frames_paths, 'Saving predictions ({}, {})'.format(model, framework)):
        
        frame_id = file_name[-8:-4]
        if frame_id in gt.keys() and axis[0]<int(frame_id)<axis[1]:
            frame = cv2.imread(file_name)
            gt_frame = np.array(dict_to_list(gt[frame_id], False))

            if segmen is not None:
                pos = np.where(segmen[:, :, 0])
                frame[pos + (np.zeros(pos[0].shape, dtype=np.uint64),)] = 255
                frame[pos + (np.ones(pos[0].shape, dtype=np.uint64),)] = 191
                frame[pos + (np.ones(pos[0].shape, dtype=np.uint64) * 2,)] = 0

            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = draw_bboxes(img, gt_frame, (0, 255, 0))

            if frame_id in dets.keys():
                dets_frame = np.array(dict_to_list(dets[frame_id], False))
                mean, std = compute_miou(gt_frame, dets_frame)

                img = draw_bboxes(img, dets_frame, (0, 0, 255))
            else:
                mean, std = 0, 0

            miou = np.hstack((miou, mean))
            std_iou = np.hstack((std_iou, std))
            plt.figure(figsize=(5, 6))

            plt.subplot(2, 1, 1)
            img = cv2.resize(img, (int(img.shape[1]*.5),int(img.shape[0]*.5)))
            plt.imshow(img)
            plt.plot(0, 0, "-", c=(0, 1, 0), label='Ground Truth')
            plt.plot(0, 0, "-", c=(0, 0, 1), label='Detection')
            plt.legend(prop={'size': 8}, loc='lower right')

            xaxis = np.hstack((xaxis, int(frame_id)))

            plt.subplot(2, 1, 2)
            plt.plot(xaxis, miou, 'cadetblue', label='Mean IoU')
            plt.fill(np.append(xaxis, xaxis[::-1]), np.append(miou + std_iou, (miou - std_iou)[::-1]), 'powderblue',
                    label='STD IoU')
            plt.axis([axis[0], axis[1], 0, 1])
            plt.xlabel('Frame id', fontsize=10)
            plt.ylabel('IoU', fontsize=10)
            plt.legend(prop={'size': 8}, loc='lower right')

            save_path = join(output_path, mode, model, framework)

            os.makedirs(save_path, exist_ok=True)

            plt.savefig(join(save_path, frame_id + '.png'))
            plt.close()


def update(i, fig, ax):
    """
    Update angle to create the animation effect on the gif plot
    """
    ax.view_init(elev=20., azim=i)
    return fig, ax

def plot_3d_surface(Xs, Ys, Zs, save_gif=False, Ztitle='mAP50'):
    """
    Plots a 3d surface from non-linear 3d-points 
    :param Xs: list with X Coords
    :param Ys: list with Y Coords
    :param Zs: list with Z Coords
    """
    fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_trisurf(Xs, Ys, Zs, cmap=cm.jet, linewidth=0)
    fig.colorbar(surf)

    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_locator(MaxNLocator(6))
    ax.zaxis.set_major_locator(MaxNLocator(5))

    fig.tight_layout()

    ax.set_xlabel('conf thres')
    ax.set_ylabel('iou thres')
    fig.savefig(Ztitle+'.png')

    if save_gif:
        anim = FuncAnimation(fig, update, frames=np.arange(0, 360, 2), repeat=True, fargs=(fig, ax))
        anim.save(Ztitle+'.gif', dpi=80, writer='imagemagick', fps=10)

def OF_plot_metrics(dif, fig_name):
    """
    Plot the difference between Optical Flow from the GT and the Estimated.

    :param dif: difference between the GT and the estimated OF
    """
    ax = plt.gca()
    im = ax.imshow(dif)
    plt.title('Optical Flow Error')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)

    plt.savefig(fig_name)
    plt.close()


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
               M[::step, ::step], scale_units='xy', angles='xy', scale=.2, color=(1, 0, 0, 1))
    plt.axis('off')
    plt.savefig(fname_output)
    plt.close()


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
        flow_hsv[:, :, 1] = flow_hsv[:, :, 1] * 2
        flow_hsv[:, :, 2] = flow_hsv[:, :, 2] * 5
        flow_color = cv2.cvtColor(flow_hsv, cv2.COLOR_HSV2RGB)

    cv2.imwrite(fname_output, flow_color)
