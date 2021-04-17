import sys
sys.path.insert(1, './AIC2018')

import os
import pickle
import numpy as np

from AIC2018.ReID.Post_tracking import parse_tracks, filter_tracks, extract_features
from AIC2018.ReID.MCT import import_pkl, remove, debug_loc, debug_id, debug_frame, dump_imgs,\
                             cluster_fill, multi_camera_matching

def _post_tracking(args):
    """
        This function performs post iou trackin operations i.e. cleans a little
        bit the tracks.

        It generates new pkl and csv files that will be used on the later stages
        of the algorithm

    """

    available_csv_files = os.listdir(args.tracking_csv)

    for csv_file in available_csv_files:
        # Read tracks
        tracks = parse_tracks(os.path.join(args.tracking_csv, csv_file))

        # Filter tracks
        tracks = filter_tracks(tracks, args.size_th, args.mask)
    
        # Extract images
        # tracks = extract_images(tracks, args.video, args.size_th, args.dist_th, args.mask, args.img_dir)
        
        if tracks is None: 
            sys.exit()
        
        # Save track obj
        save_pkl_path = os.path.join(args.output_path, 'pkl')
        
        os.makedirs(save_pkl_path, exist_ok=True)
        # os.system('mkdir -p %s' % args.output)
        
        file_name = csv_file.split('.')[0]

        with open(os.path.join(save_pkl_path,  '%s.pkl'%file_name), 'wb') as f:
            pickle.dump(tracks, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        dets = []
        for t in tracks:
            dets.append(t.dump())
        dets = np.concatenate(dets, axis=0)

        save_csv_path = os.path.join(args.output_path, 'csv')
        os.makedirs(save_csv_path, exist_ok=True)
        np.savetxt(os.path.join(save_csv_path, '%s.csv'%file_name), dets, delimiter=',', fmt='%f')


def _multi_camera_tracking(args):
    """
    Args:
        args:

    Returns:

    """

    # Create sequence names
    #locs = os.listdir(args.data_path)
    locs = ['S01','S03']
    locs.sort()
    model_path = os.path.join(os.getcwd(), args.data_path.split('/AICity')[0], 'AIC2018_models/model_880_base.ckpt')
    # loc_n_seq = ['001', 6, 2, 3]

    # Load and initialize tracks objs with seq names
    multi_cam_tracks = []
    seq_id = 1
    loc_seq_id = []
    for i, l in enumerate(locs):
        single_cam_tracks = []
        seqs = []
        loc_n_seq = os.listdir(os.path.join(args.data_path, l))
        for n in loc_n_seq:
            img_dir = os.path.join(args.data_path, l, n, 'vdo')
            pkl_name = os.path.join(args.output_path, 'pkl', n + '.csv')
            tracks = import_pkl(pkl_name.split('.csv')[0] + '.pkl')
            tracks = extract_features(tracks, img_dir, model_path, args.n_layers, args.batch_size)

            for t in tracks:
                t.assign_seq_id(seq_id, int(l[-1]))
            seqs.append(seq_id)

            seq_id += 1
            single_cam_tracks += tracks
        loc_seq_id.append(seqs)
        multi_cam_tracks.append(single_cam_tracks)

    # Multi camera matching
    # len of multi_cam_tracks is equal to the number of Locations
    tracks = multi_camera_matching(args, multi_cam_tracks, locs)
    with open(os.path.join(args.output_path, 'after_mct.pkl'), 'wb') as f:
        pickle.dump(tracks, f)
    # with open(os.path.join(args.output_dir, 'after_mct.pkl'), 'rb') as f:
    #    tracks = pickle.load(f)
    # print(len(tracks))
    # sys.exit()

    # Remove detections with known submissions
    if args.filter is not None:
        tracks = remove(args, tracks)
        with open(os.path.join(args.output_path, 'after_remove.pkl'), 'wb') as f:
            pickle.dump(tracks, f)

    # Decide the final 100 tracks
    # tracks = sample_tracks(tracks, 100)
    # tracks = sample_tracks(tracks, 300)[200:300]
    # tracks = fill(tracks, 100)
    # tracks = cluster_fill(args, tracks, 1)

    # Re-index id & final check
    # for i, t in enumerate(tracks):
    #     t.assign_id(i + 1)
    #     if not debug_loc(t.dump(), loc_seq_id):
    #         sys.exit('Does not satisfy location condition!')
    #     if not debug_frame(t.dump()):
    #         sys.exit('Does not satisfy frame condition!')
    #     if not debug_id(t.dump()):
    #         sys.exit('Does not satisfy object id condition')

    # Output to file
    dets = []
    for t in tracks:
        dets.append(t.dump())
    dets = np.concatenate(dets, axis=0)
    dets = np.concatenate([dets[:, :7], -1 * np.ones((dets.shape[0], 1)), dets[:, [7]]], axis=1)
    dets[:, 5:7] = dets[:, 5:7] + dets[:, 3:5]
    np.savetxt(os.path.join(args.output_path, 'track3.txt'),
               dets, fmt='%d %d %d %d %d %d %d %d %f')

    # Dump imgs
    print('dumping images...')
    os.makedirs(args.dump_dir, exist_ok=True)
    for t in tracks:
        dump_imgs(args.dump_dir, t)


def compute_multitracking(args):
    """
        This function implements the full pipeline for AIC2018
    """

    _post_tracking(args)
    _multi_camera_tracking(args)
