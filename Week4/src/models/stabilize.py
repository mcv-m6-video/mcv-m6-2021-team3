import cv2
import os
import numpy as np
from tqdm.auto import tqdm
from os.path import join, exists
from utils.utils import pol2cart, write_png_flow, read_kitti_OF
from models.optical_flow import block_matching, smoothing, fixBorder

def seq_stabilization_BM(frames_paths, output_path, bm_opts):
    #resize to sped up OF
    H, W, C = cv2.imread(frames_paths[0]).shape
    dsize = (W,H)#(int(W*0.25), int(H*0.25))

    os.makedirs(join(output_path, 'bm'),exist_ok=True)
    os.makedirs(join(output_path, 'seq_stabilization'),exist_ok=True)
    
    png_name = join(output_path, 'bm', '_'.join(('ws-'+str(bm_opts['window_size']),
                                                 'shift-'+str(bm_opts['shift']),
                                                 'stride-'+str(bm_opts['stride']),
                                                 'f_id.png')))

    for f_id, (f1_path, f2_path) in enumerate(tqdm(zip(frames_paths[:-1],frames_paths[1:]), 'Stabilization in progress')):
        #load pair of frames
        img1 = cv2.resize(cv2.imread(f1_path), dsize)
        img2 = cv2.resize(cv2.imread(f2_path), dsize)

        #OF
        if not exists(png_name.replace('f_id',str(f_id))):
            pred_OF = block_matching(img1, img2, bm_opts['window_size'], bm_opts['shift'], bm_opts['stride'])
            write_png_flow(pred_OF, png_name.replace('f_id',str(f_id)))
        else:
            pred_OF = read_kitti_OF(png_name.replace('f_id',str(f_id)))

        occ = pred_OF[:, :, 2]
        u = pred_OF[:, :, 0] * occ
        v = pred_OF[:, :, 1] * occ
        mag, ang = cv2.cartToPolar(u.astype(np.float32), v.astype(np.float32))
        #keep the values which is foudn the most for mag and ang
        uniques, counts = np.unique(mag, return_counts=True)
        mc_mag = uniques[counts.argmax()]
        uniques, counts = np.unique(ang, return_counts=True)
        mc_ang = uniques[counts.argmax()]
        u, v = pol2cart(mc_mag, mc_ang)
        #Create an affine transformation for v and u
        affine_H = np.float32([[1, 0, -v],[0,1,-u]])
        #Compute affine transforamtion
        img2_stabilized = cv2.warpAffine(img2,affine_H,(img2.shape[1],img2.shape[0]))
        cv2.imwrite(join(output_path,'seq_stabilization','%04d' % f_id +'.png'),img2_stabilized)


def seq_stabilization_LK(frames_paths, output_path):
    
    os.makedirs(join(output_path, 'opencv_stabilization'),exist_ok=True)

    n_frames = int(len(frames_paths))
    H, W, C = cv2.imread(frames_paths[0]).shape
    dsize = (int(W*0.25), int(H*0.25))
    transforms = np.zeros((n_frames-1, 3), np.float32) 
    prev = cv2.imread(frames_paths[0])
    prev = cv2.resize(prev, dsize)
    prev = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

    for f_id, frame_path in enumerate(tqdm(frames_paths[1:], 'OF and transformation computation')):

        #load pair of frames
        curr = cv2.imread(frame_path) 
        curr = cv2.resize(curr, dsize) 
        curr = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
        #OF computation
        prev_pts = cv2.goodFeaturesToTrack(prev, maxCorners=200, qualityLevel=0.01, minDistance=30, blockSize=3)
        curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev, curr, prev_pts, None)
        idx = np.where(status==1)[0]
        prev_pts = prev_pts[idx]
        curr_pts = curr_pts[idx]
        assert prev_pts.shape == curr_pts.shape
            
        #Tranformation estimation
        m, _ = cv2.estimateAffinePartial2D(prev_pts, curr_pts) 
        dx = m[0,2]
        dy = m[1,2]
        da = np.arctan2(m[1,0], m[0,0]) # Extract rotation angle
        transforms[f_id] = [dx,dy,da] 

        prev = curr
        
    # Find the cumulative sum of tranform matrix for each dx,dy and da
    trajectory = np.cumsum(transforms, axis=0) 

    smoothed_trajectory = smoothing(trajectory, 50)
    difference = smoothed_trajectory - trajectory
    transforms_smooth = transforms + difference

    for f_id, path in enumerate(tqdm(frames_paths[0:-1], 'Stabilization in process')):

        frame = cv2.imread(frames_paths[f_id]) 
        frame = cv2.resize(frame, dsize)
        # Extract transformations from the new transformation array
        dx = transforms_smooth[f_id,0]
        dy = transforms_smooth[f_id,1]
        da = transforms_smooth[f_id,2]
        # Reconstruct transformation matrix accordingly to new values
        m = np.zeros((2,3), np.float32)
        m[0,0] = np.cos(da)
        m[0,1] = -np.sin(da)
        m[1,0] = np.sin(da)
        m[1,1] = np.cos(da)
        m[0,2] = dx
        m[1,2] = dy
        # Apply affine wrapping to the given frame
        frame_stabilized = cv2.warpAffine(frame, m, (W,H))

        # Fix border artifacts
        frame_stabilized = fixBorder(frame_stabilized)
        cv2.imshow('Frame', frame_stabilized)
        cv2.waitKey(1)
        cv2.imwrite(join(output_path,'opencv_stabilization','%04d' % f_id +'.png'),frame_stabilized)
