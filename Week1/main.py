import sys
from metrics import *
from visualize import *
from utils import *

display = True
noise_evaluation = True

path = '../'
visualization = 'base'


def main(argv):
    if len(argv) > 1:
        task = float(argv[1])
    else:
        task = 3

    if int(task) in [1, 2]:
        data_dir = join(path, 'AICity_data/train/S03/c010')
        gt_dir = join(data_dir, 'gt')
        vdo_dir = join(data_dir, 'vdo')

        # first load gt
        gt = load_annot(path, 'ai_challenge_s03_c010-full_annotation.xml')  # load_annot(gt_dir,'gt.txt')
        imagenames = load_frames(vdo_dir)

        if task == 1.1:
            # T1.1: IoU & mAP (GT + noise)
            # Create noisy detections

            # Report on noise effects
            if noise_evaluation:
                gen_eval = gen_noise_eval(imagenames, gt)
                plot_metrics_noise(gen_eval)
            # Single evaluation
            else:
                dets = {}
                for frame, info in gt.items():
                    gen_bbox = gen_noisy_bbox(dict_to_list(info), bbox_generate=True)

                    for bbox in gen_bbox:
                        dets = update_data(dets, int(frame), bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3], 1.)

                _, _, ap = voc_eval(gt, imagenames, dets)
                if display:
                    print('AP50:', ap)

        else:
            det_dir = join(data_dir, 'det')

            dets = {}
            det_models = os.listdir(det_dir)
            for det_model in det_models:
                # read dets
                dets.update({det_model: load_annot(det_dir, det_model)})

                if task == 1.2:
                    # T1.2: mAP for provided object detections
                    _, _, ap = voc_eval(gt, imagenames, dets[det_model])
                    if display:
                        print(det_model.split('.')[0], '--> AP50:', ap)

                if task == 2:
                    # T2: IoU vs time
                    visualize_iou(gt,dets[det_model],imagenames,det_model)

    elif task in [3, 4]:

        gt_dir = join(path, 'data_stereo_flow/training/flow_noc')
        det_dir = join(path, 'results_opticalflow_kitti/results')
        img_dir = join(path, 'data_stereo_flow/training/colored_0')

        OF, GTOF, DIF, MSEN, PEPN, OF_occ = {}, {}, {}, {}, {}, {}
        for of_path in os.listdir(det_dir):
            if 'png' not in of_path:
                continue
            seq = (of_path.split('.')[0]).replace('LKflow_', '')

            GTOF.update({seq: read_kitti_OF(join(gt_dir, seq + '.png'))})
            OF.update({seq: read_kitti_OF(join(det_dir, of_path))})
            OF_occ.update({seq: read_kitti_OF(join(det_dir, of_path))})

            occluded_idx = GTOF[seq][:, :, 2] == 0
            OF_occ[seq][occluded_idx, :] = 0

            if task == 3:
                # T3.1: MSEN & T3.2 PEPN

                # set occluded pixels to 0
                occluded_idx = GTOF[seq][:, :, 2] == 0
                OF[seq][occluded_idx, :] = 0

                DIF.update({seq: vec_error(GTOF[seq], OF[seq], 2)})
                MSEN.update({seq: compute_error(gt=GTOF[seq], error=DIF[seq][0])})
                PEPN.update({seq: compute_error(gt=GTOF[seq], error=DIF[seq][0], op='pep')})

                mean_error = np.mean(DIF[seq][1])
                std_error = np.std(DIF[seq][1])

                if display:
                    plot_metrics_OF(seq, GTOF, OF, DIF, mean_error, std_error)

            if task == 4:
                # T4: Optical flow plot

                OF_arr = OF[seq]
                GTOF_arr = GTOF[seq]
                OF_occ_arr = OF_occ[seq]
                imgOF = cv2.imread(join(img_dir, seq + '.png'), 1)

                os.makedirs('task4',exist_ok=True)

                if visualization == 'base':
                    dif = OF_arr - OF_occ_arr

                    OF_quiver_visualize(imgOF, GTOF_arr, step=15, fname_output='task4/flow_gt_' + seq + '_quiver.png')
                    OF_quiver_visualize(imgOF, OF_arr, step=8, fname_output='task4/flow_det_' + seq + '_quiver.png')
                    OF_quiver_visualize(imgOF, OF_occ_arr, step=8, fname_output='task4/flow_det_occ_' + seq + '_quiver.png')
                    OF_quiver_visualize(imgOF, dif, step=8, fname_output='task4/flow_dif_' + seq + '_quiver.png')

                elif visualization == 'hsv':
                    OF_hsv_visualize(GTOF_arr, fname_output='task4/flow_gt_' + seq + '_hsv.png', enhance=True)
                    OF_hsv_visualize(OF_arr, fname_output='task4/flow_det_' + seq + '_hsv.png', enhance=True)

                elif visualization == 'color_wheel':
                    OF_colorwheel_visualize(GTOF_arr, fname_output='task4/flow_gt_' + seq + '_color_wheel.png', enhance=True)
                    OF_colorwheel_visualize(OF_arr, fname_output='task4/flow_det_' + seq + '_color_wheel.png', enhance=True)


if __name__ == "__main__":
    main(sys.argv)
