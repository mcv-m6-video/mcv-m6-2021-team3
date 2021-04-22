import sys
sys.path.insert(1, './AIC2018')

import os
from os.path import dirname, join, exists
import pickle
import numpy as np
from tqdm.auto import tqdm
from sklearn.utils import shuffle
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.mixture import GaussianMixture
from scipy.spatial.distance import braycurtis
from itertools import combinations, product
from scipy.optimize import linear_sum_assignment as linear_assignment

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from utils.utils import dict_to_array, color_hist, str_frame_id, write_json_file, read_json_file, match_trajectories, array_to_dict
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


def iamai_multitracking(args):
    """
        This function implements the full pipeline for AIC2018
    """

    _post_tracking(args)
    _multi_camera_tracking(args)

def hist_multitracking(det_bboxes, frames_path, opts):

    # Get config
    bins = opts.bins
    color_space = opts.color_space
    cluster = opts.cluster_hist
    hist_file = opts.hist_file

    # Define new dictionary of detections
    new_det_bboxes = det_bboxes.copy()

    # Define color spaces used and their channel ranges
    map_color_spaces = {cs:i for i,cs in enumerate(['rgb','hsv','lab','ycrcb'])}
    COLOR_SPACES = ['cv2.COLOR_BGR2RGB', 'cv2.COLOR_BGR2HSV','cv2.COLOR_BGR2LAB','cv2.COLOR_BGR2YCR_CB']
    COLOR_RANGES = [[[0,255]]*3, [[0,179]]+[[0,255]]*2, [[0,255]]*3, [[0,255]]*3]

    if not exists(hist_file):
        # Initialize histogram and dict of arrays of detections
        mean_hist={}
        det_array={}

        for cam, dets in tqdm(det_bboxes.items(),'Computing color histograms'):
            # Update dict with array of dets for the camera
            det_array.update({cam:dict_to_array(dets)})
            # Get image directory
            img_dir = dirname(frames_path[cam][0])

            # Define frames and detections ids
            frames_ids = np.unique(det_array[cam][:,0])
            det_ids = np.unique(det_array[cam][:, 1])
            n_dets = len(det_ids)

            # Initialize histogram (num of detections, num color spaces, bins, channels)
            hist = np.empty([det_array[cam].shape[0],len(COLOR_SPACES),bins,3],dtype=float)
            
            # For loop for each frame with detections
            for frame_id in frames_ids:
                # Get detections at that frame and compute histogram of each
                loc = np.where(det_array[cam][:,0] == frame_id)[0]
                hist[loc,:,:,:] = color_hist(join(img_dir,str_frame_id(frame_id)+'.jpg'), det_array[cam][loc, 2:6].astype(int), COLOR_SPACES, COLOR_RANGES, bins)
            
            # Organize histogram by detections ids
            trajs = [hist[np.where(det_array[cam][:, 1] == det_ids[i])[0], :]
                        for i in range(n_dets)]
            
            # Update histogram for the camera
            mean_hist.update({cam:[np.mean(np.mean(traj,axis=0),axis=2) for traj in trajs]})
        
        # Prepare and save results
        mean_hist_json = {k:[d.tolist() for d in data] for k, data in mean_hist.items()}
        write_json_file(mean_hist_json, hist_file)

    else:
        # Read results of histogram
        mean_hist=read_json_file(hist_file)
        
        # Create dictionary of arrays
        det_array={}
        for cam, dets in det_bboxes.items():
            det_array.update({cam:dict_to_array(dets)})
    
    
    if cluster in ['kmeans','gmm']:
        #Define number of clusters equal to max number of ids
        k = np.max([len(m_h) for m_h in mean_hist.values()])

        # Stack all data into a matrix
        data = np.vstack([np.stack(hist) for hist in mean_hist.values()])
        cam_id = np.vstack([i*np.ones((np.stack(val).shape[0],1)) for i,val in enumerate(mean_hist.values())])
        
        # Choose color space
        c = map_color_spaces[color_space]
            
        if cluster in 'kmeans':
            # Kmeans clustering
            kmeans = KMeans(n_clusters=k, random_state=0).fit(data[:,c,:])
            labels = kmeans.labels_

        elif cluster in 'gmm':
            # Gaussian mixture model clustering
            labels_ = GaussianMixture(n_components=k, random_state=0).fit_predict(data[:,c,:])
            labels = labels_

        # LDA or PCA to visualize
        if opts.vis_cluster:
            lda = LDA()
            data_ = lda.fit_transform(data[:,c,:],labels)

            #pca = PCA(3) 
            #data_ = pca.fit_transform(data[:,c,:])

            fig = plt.figure(figsize=(6,5))
            ax = fig.add_subplot(projection='3d')
            ax.scatter(data_[:,0],data_[:,1],data_[:,2],c=labels)
            plt.savefig(hist_file.replace('json','png'))
        
        # Split results per camera
        res_ids = [labels[np.where(cam_id==i)[0]] for i in np.unique(cam_id)]

        # Save results to dict
        for ids, (cam, array) in zip(res_ids, det_array.items()):            
            for prev_id, new_id in zip(np.unique(array[:,1]), ids):
                array[np.where(array[:,1]==prev_id)[0],1] = new_id
            new_det_bboxes.update({cam:array_to_dict(array)})
    

    elif cluster in 'braycurtis':
        ## MATCH IDS USING BRAY CURTIS DISTANCE ##
        
        # Get the id of the camera with large number of detection ids
        max_ids = np.argmax([len(m_h) for m_h in mean_hist.values()])

        # Define mapper form int to str cam
        cam_map = {c:cam for c,cam in enumerate(mean_hist)}

        # Create combinations of the camera with larger ids and the rest
        cam_comb = list(product([cam_map[max_ids]],[cam for cam in mean_hist if cam not in cam_map[max_ids]]))

        new_det_bboxes.update({cam_map[max_ids]:det_bboxes[cam_map[max_ids]]})
        # For loop over the combinations
        for (cam1, cam2) in cam_comb:
            # Initialize bray curtis distance matrix
            bc_distance = np.empty((len(mean_hist[cam1]),len(mean_hist[cam2]),len(COLOR_SPACES)))
            
            # For every pair of car trajectories compute distance
            for (t1, t2) in list(product(list(range(len(mean_hist[cam1]))),list(range(0,len(mean_hist[cam2]))))):
                for c in range(len(COLOR_SPACES)):
                    bc_distance[t1,t2,c] = braycurtis(np.array(mean_hist[cam1][t1])[c,:], np.array(mean_hist[cam2][t2])[c,:], w=None)
            
            # Find matches and re-assign ids
            matches = linear_assignment(np.sum(bc_distance,axis=2))
            det_array[cam2] = match_trajectories(det_array[cam2],matches,[np.unique(det_array[cam1][:, 1]),np.unique(det_array[cam2][:, 1])])
            new_det_bboxes.update({cam2:array_to_dict(det_array[cam2])})
    
    return new_det_bboxes

