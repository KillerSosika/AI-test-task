import torch
import numpy as np
import cv2

def generate_loftr_supervision(img0_np, img1_np, coarse_scale=8):
    h, w = img0_np.shape[:2]
    hc, wc = h // coarse_scale, w // coarse_scale
    L = hc * wc  # Total number of patches in the grid

    sift = cv2.SIFT_create(nfeatures=2000)
    kp0, des0 = sift.detectAndCompute(img0_np, None)
    kp1, des1 = sift.detectAndCompute(img1_np, None)

    # Correct GT matrix of shape (L, L)
    conf_matrix_gt = torch.zeros((L, L), dtype=torch.float32)

    if kp0 and kp1 and des0 is not None and des1 is not None:
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        matches = bf.knnMatch(des0, des1, k=2)
        
        good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]

        for m in good_matches:
            pt0, pt1 = kp0[m.queryIdx].pt, kp1[m.trainIdx].pt

            cx0, cy0 = int(pt0[0] // coarse_scale), int(pt0[1] // coarse_scale)
            cx1, cy1 = int(pt1[0] // coarse_scale), int(pt1[1] // coarse_scale)

            if 0 <= cx0 < wc and 0 <= cy0 < hc and 0 <= cx1 < wc and 0 <= cy1 < hc:
                # Convert 2D coordinates to 1D indices
                idx0 = cy0 * wc + cx0
                idx1 = cy1 * wc + cx1
                
                # Set 1.0 at the intersection of the matching patches
                conf_matrix_gt[idx0, idx1] = 1.0

    return {
        "spv_conf_matrix_gt": conf_matrix_gt
    }