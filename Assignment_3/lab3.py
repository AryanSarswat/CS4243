import numpy as np
from skimage import filters
from skimage.feature import corner_peaks
from scipy.spatial.distance import cdist
from scipy.ndimage.filters import convolve
from scipy.ndimage import gaussian_filter
import math


### REMOVE THIS
from cv2 import findHomography

from utils import pad, unpad

import cv2
_COLOR_RED = (255, 0, 0)
_COLOR_GREEN = (0, 255, 0)
_COLOR_BLUE = (0, 0, 255)

_COLOR_RED = (255, 0, 0)
_COLOR_GREEN = (0, 255, 0)
_COLOR_BLUE = (0, 0, 255)

def trim(frame):
    if not np.sum(frame[0]):
        return trim(frame[1:])
    if not np.sum(frame[-1]):
        return trim(frame[:-2])
    if not np.sum(frame[:,0]):
        return trim(frame[:,1:])
    if not np.sum(frame[:,-1]):
        return trim(frame[:,:-2])
    return frame

##################### PART 1 ###################

# 1.1 IMPLEMENT
def harris_corners(img, window_size=3, k=0.04):
    '''
    Compute Harris corner response map. Follow the math equation
    R=Det(M)-k(Trace(M)^2).

    Hint:
        You may use the functions filters.sobel_v filters.sobel_h & scipy.ndimage.filters.convolve, 
        which are already imported above
        
    Args:
        img: Grayscale image of shape (H, W)
        window_size: size of the window function
        k: sensitivity parameter

    Returns:
        response: Harris response image of shape (H, W)
    '''

    H, W= img.shape
    window = np.ones((window_size, window_size))
    response = np.zeros((H, W))

    # YOUR CODE HERE
    Ix = filters.sobel_v(img)
    Iy = filters.sobel_h(img)
    
    Ix2 = Ix * Ix
    Iy2 = Iy * Iy
    IxIy = Ix * Iy
    
    SummedIx2 = convolve(Ix2, np.ones((window_size,window_size)))
    SummedIy2 = convolve(Iy2, np.ones((window_size,window_size)))
    SummedIxIy = convolve(IxIy, np.ones((window_size,window_size)))

    shape = SummedIx2.shape
    response = np.zeros(shape)

    for row in range(shape[0]):
        for col in range(shape[1]):
            H = np.array([[SummedIx2[row][col],SummedIxIy[row][col]],[SummedIxIy[row][col],SummedIy2[row][col]]])
            response[row][col] = np.linalg.det(H) - k * (np.trace(H) ** 2)
    # END        
    return response

# 1.2 IMPLEMENT
def naive_descriptor(patch):
    '''
    Describe the patch by normalizing the image values into a standard 
    normal distribution (having mean of 0 and standard deviation of 1) 
    and then flattening into a 1D array. 
    
    The normalization will make the descriptor more robust to change 
    in lighting condition.

    Args:
        patch: grayscale image patch of shape (h, w)
    
    Returns:
        feature: 1D array of shape (h * w)
    '''
    feature = []
    ### YOUR CODE HERE
    mu = np.mean(patch)
    sigma = np.std(patch)
    norm = patch - mu / (sigma + 0.0001)
    norm = np.where(norm > 0, norm, 0)
    feature = np.ndarray.flatten(norm)
    ### END YOUR CODE

    return feature

# GIVEN
def describe_keypoints(image, keypoints, desc_func, patch_size=16):
    '''
    Args:
        image: grayscale image of shape (H, W)
        keypoints: 2D array containing a keypoint (x, y) in each row
        desc_func: function that takes in an image patch and outputs
            a 1D feature vector describing the patch
        patch_size: size of a square patch at each keypoint
                
    Returns:
        desc: array of features describing the keypoints
    '''

    image.astype(np.float32)
    desc = []
    for i, kp in enumerate(keypoints):
        y, x = kp
        patch = image[np.max([0,y-(patch_size//2)]):y+((patch_size+1)//2),
                      np.max([0,x-(patch_size//2)]):x+((patch_size+1)//2)]
      
        desc.append(desc_func(patch))
   
    return np.array(desc)

# GIVEN
def make_gaussian_kernel(ksize, sigma):
    '''
    Good old Gaussian kernel.
    :param ksize: int
    :param sigma: float
    :return kernel: numpy.ndarray of shape (ksize, ksize)
    '''

    ax = np.linspace(-(ksize - 1) / 2., (ksize - 1) / 2., ksize)
    yy, xx = np.meshgrid(ax, ax)

    kernel = np.exp(-0.5 * (np.square(yy) + np.square(xx)) / np.square(sigma))

    return kernel / kernel.sum()


# 1.2 IMPLEMENT
def simple_sift(patch):
    '''
    Your implementation does not need to exactly match the SIFT reference.
    Here are the key properties your (baseline) descriptor should have:
    (1) a 4x4 grid of cells, each length of 16/4=4. It is simply the
        terminology used in the feature literature to describe the spatial
        bins where gradient distributions will be described.
    (2) each cell should have a histogram of the local distribution of
        gradients in 8 orientations. Appending these histograms together will
        give you 4x4 x 8 = 128 dimensions.
    (3) Each feature should be normalized to unit length.

    Use the gradient orientation to determine the bin, and the gradient magnitude * weight from
    the Gaussian kernel as vote weight.

    Args:
        patch: grayscale image patch of shape (h, w)

    Returns:
        feature: 1D array of shape (128, )
    '''
    
    # You can change the parameter sigma, which has been default to 3
    weights = np.flipud(np.fliplr(make_gaussian_kernel(patch.shape[0],3)))
    
    histogram = np.zeros((4,4,8))
    
    # YOUR CODE HERE
    Ix = filters.sobel_v(patch)
    Iy = filters.sobel_h(patch)
    grad_mag = np.sqrt((Ix**2) + (Iy**2))
    grad_angle = np.arctan2(Iy,Ix) * 180/np.pi
    grad_mag = grad_mag * weights

    shape = patch.shape
    hists = []

    for i in range(shape[0]//4):
        for j in range(shape[1]//4):
            grad_window = grad_angle[i * 4: (i * 4) + 4, j * 4 : (j * 4) + 4]
            weight_window = grad_mag[i * 4: (i * 4) + 4, j * 4 : (j * 4) + 4]

            temp_hist = np.zeros(8)

            for row in range(4):
                for col in range(4):
                    angle = grad_window[row][col]
                    weight = weight_window[row][col]

                    if 0 <= angle < 45:
                        temp_hist[0] += weight
                    elif 45 <= angle < 90:
                        temp_hist[1] += weight
                    elif 90 <= angle < 135:
                        temp_hist[2] += weight
                    elif 135 <= angle <= 180:
                        temp_hist[3] += weight
                    elif -45 <= angle < 0:
                        temp_hist[7] += weight
                    elif -90 <= angle < -45:
                        temp_hist[6] += weight
                    elif -135 <= angle < -90:
                        temp_hist[5] += weight
                    elif -180 <= angle < -135:
                        temp_hist[4] += weight
                        
            hists.append(temp_hist)
    hists = np.array(hists)
    feature = hists.flatten()
    feature = feature / np.linalg.norm(feature)
    # END
    return feature

# 1.3 IMPLEMENT
def top_k_matches(desc1, desc2, k=2):
    '''
    Compute the Euclidean distance between each descriptor in desc1 versus all descriptors in desc2 (Hint: use cdist).
    For each descriptor Di in desc1, pick out k nearest descriptors from desc2, as well as the distances themselves.
    Example of an output of this function:
    
        [(0, [(18, 0.11414082134194799), (28, 0.139670625444803)]),
         (1, [(2, 0.14780585099287238), (9, 0.15420019834435536)]),
         (2, [(64, 0.12429203239414029), (267, 0.1395765079352806)]),
         ...<truncated>
    '''
    match_pairs = []
    # YOUR CODE HERE
    diff = cdist(desc1, desc2)
    for i in range(len(diff)):
        dist = diff[i]
        sort_dist = sorted(dist)[:k]
        temp_tup = (i , [])
        for j in sort_dist:
            temp_tup[1].append((np.where(dist == j)[0], j))
        match_pairs.append(temp_tup)
    # END
    return match_pairs

# 1.3 IMPLEMENT
def ratio_test_match(desc1, desc2, match_threshold):
    '''
    Match two set of descriptors using the ratio test.
    Output should be a numpy array of shape (k,2), where k is the number of matches found. 
    In the following sample output:
        array([[  3,   0],
               [  5,  30],
               [ 11,   9],
               [ 18,   7],
               [ 24,   5],
               [ 30,  17],
               [ 32,  24],
               [ 46,  23], ... <truncated>
              )
              
        desc1[3] is matched with desc2[0], desc1[5] is matched with desc2[30], and so on.
    
    All other match functions will return in the same format as does this one.
    
    '''
    match_pairs = []
    top_2_matches = top_k_matches(desc1, desc2)
    # YOUR CODE HERE
    for i in top_2_matches:
        if (i[1][0][1] / i[1][1][1])  < match_threshold:
            match_pairs.append([i[0], int(i[1][0][0])])  # Without cast 2nd element is array object
    # END
    # Modify this line as you wish
    match_pairs = np.array(match_pairs)
    return match_pairs

# GIVEN
def compute_cv2_descriptor(im, method=cv2.SIFT_create()):
    '''
    Detects and computes keypoints using one of the implementations in OpenCV
    You can use:
        cv2.SIFT_create()

    Do note that the keypoints coordinate is (col, row)-(x,y) in OpenCV. We have changed it to (row,col)-(y,x) for you. (Consistent with out coordinate choice)
    '''
    kpts, descs = method.detectAndCompute(im, None)
    
    keypoints = np.array([(kp.pt[1],kp.pt[0]) for kp in kpts])
    angles = np.array([kp.angle for kp in kpts])
    sizes = np.array([kp.size for kp in kpts])
    
    return keypoints, descs, angles, sizes

##################### PART 2 ###################

# GIVEN
def transform_homography(src, h_matrix, getNormalized = True):
    '''
    Performs the perspective transformation of coordinates

    Args:
        src (np.ndarray): Coordinates of points to transform (N,2)
        h_matrix (np.ndarray): Homography matrix (3,3)

    Returns:
        transformed (np.ndarray): Transformed coordinates (N,2)

    '''
    transformed = None

    input_pts = np.insert(src, 2, values=1, axis=1)
    transformed = np.zeros_like(input_pts)
    transformed = h_matrix.dot(input_pts.transpose())
    if getNormalized:
        transformed = transformed[:-1]/transformed[-1]
    transformed = transformed.transpose().astype(np.float32)
    
    return transformed

# 2.1 IMPLEMENT
def compute_homography(src, dst):
    '''
    Calculates the perspective transform from at least 4 points of
    corresponding points using the **Normalized** Direct Linear Transformation
    method.

    Args:
        src (np.ndarray): Coordinates of points in the first image (N,2)
        dst (np.ndarray): Corresponding coordinates of points in the second
                          image (N,2)

    Returns:
        h_matrix (np.ndarray): The required 3x3 transformation matrix H.

    Prohibited functions:
        cv2.findHomography(), cv2.getPerspectiveTransform(),
        np.linalg.solve(), np.linalg.lstsq()
    '''
    h_matrix = np.eye(3, dtype=np.float64)
  
    # YOUR CODE HERE
    num_points = src.shape[0]
    src_x = src[:, 0]
    src_y = src[:, 1]
    dst_x = dst[:, 0]
    dst_y = dst[:, 1]

    # Normalization matrix: Translate by mean vector & scale by SD/sqrt(2)
    src_mx = np.mean(src_x)
    src_my = np.mean(src_y)
    src_sx = np.std(src_x) / np.sqrt(2)
    src_sy = np.std(src_y) / np.sqrt(2)
    T_src = np.array([[1/src_sx, 0, -src_mx/src_sx], [0, 1/src_sy, -src_my/src_sy], [0, 0, 1]])

    dst_mx = np.mean(dst_x)
    dst_my = np.mean(dst_y)
    dst_sx = np.std(dst_x) / np.sqrt(2)
    dst_sy = np.std(dst_y) / np.sqrt(2)
    T_dst = np.array([[1/dst_sx, 0, -dst_mx/dst_sx], [0, 1/dst_sy, -dst_my/dst_sy], [0, 0, 1]])

    # Normalize src and dst
    src_norm = transform_homography(src, T_src)
    dst_norm = transform_homography(dst, T_dst)

    # DLT
    A = []
    for i in range(num_points):
        x = src_norm[i, 0]
        y = src_norm[i, 1]
        x_p = dst_norm[i, 0]
        y_p = dst_norm[i, 1]
        A.append([-x, -y, -1, 0, 0, 0, x*x_p, y*x_p, x_p])
        A.append([0, 0, 0, -x, -y, -1, x*y_p, y*y_p, y_p])

    A = np.array(A)
    u, s, vh = np.linalg.svd(A)
    H = vh[:, -1].reshape((3,3))

    # Denormalization: Revert initial transformations
    h_matrix = np.matmul(np.linalg.inv(T_dst), np.matmul(H, T_src))

    # END 

    return h_matrix

# 2.2 IMPLEMENT
def ransac_homography(keypoints1, keypoints2, matches, sampling_ratio=0.5, n_iters=500, delta=20):
    """
    Use RANSAC to find a robust affine transformation

        1. Select random set of matches
        2. Compute affine transformation matrix
        3. Compute inliers
        4. Keep the largest set of inliers
        5. Re-compute least-squares estimate on all of the inliers

    Args:
        keypoints1: M1 x 2 matrix, each row is a point
        keypoints2: M2 x 2 matrix, each row is a point
        matches: N x 2 matrix, each row represents a match
            [index of keypoint1, index of keypoint 2]
        sampling_ratio: percentage of points selected at each iteration
        n_iters: the number of iterations RANSAC will run
        threshold: the threshold to find inliers

    Returns:
        H: a robust estimation of affine transformation from keypoints2 to
        keypoints 1
    """
    N = matches.shape[0]
    n_samples = int(N * sampling_ratio)

    matched1_unpad = keypoints1[matches[:,0]]
    matched2_unpad = keypoints2[matches[:,1]]

    max_inliers = np.zeros(N)
    n_inliers = 0

    # RANSAC iteration start
    ### YOUR CODE HERE
    H = np.eye(3, dtype=np.float64)
    for i in range(n_iters):
        curr_inliers = []
        num_inliers = 0

        # Get random samples
        indices = np.random.randint(N, size=n_samples)
        samples = matches[indices]
        src = keypoints1[samples[:, 0]]
        dst = keypoints2[samples[:, 0]]
        H_temp = compute_homography(src, dst)

        # Produce H, find inliers and update inlier stores if new max inliers found
        for i in indices:
            P = np.array([keypoints1[matches[i][0]][0], keypoints1[matches[i][0]][1], 1])
            P_match = keypoints2[matches[i][1]]
            P_res = np.matmul(H_temp, P)
            P_proj = np.array([P_res[0], P_res[1]])
            if np.linalg.norm(P_match - P_proj) <= delta:
                num_inliers += 1
                curr_inliers.append(i)

            if num_inliers > n_inliers:
                n_inliers = num_inliers
                H = H_temp
                max_inliers = np.array(curr_inliers)
    
    # Re-compute H with inliers
    inliers_val = matches[max_inliers]
    re_src = keypoints1[inliers_val[:, 0]]
    re_dst = keypoints1[inliers_val[:, 1]]
    H = compute_homography(re_src, re_dst)
    
    ### END YOUR CODE
    return H, matches[max_inliers]

##################### PART 3 ###################
# GIVEN FROM PREV LAB
from skimage.feature import peak_local_max
def find_peak_params(hspace, params_list,  window_size=1, threshold=0.5):
    '''
    Given a Hough space and a list of parameters range, compute the local peaks
    aka bins whose count is larger max_bin * threshold. The local peaks are computed
    over a space of size (2*window_size+1)^(number of parameters)

    Also include the array of values corresponding to the bins, in descending order.
    '''
    assert len(hspace.shape) == len(params_list), \
        "The Hough space dimension does not match the number of parameters"
    for i in range(len(params_list)):
        assert hspace.shape[i] == len(params_list[i]), \
            f"Parameter length does not match size of the corresponding dimension:{len(params_list[i])} vs {hspace.shape[i]}"
    peaks_indices = peak_local_max(hspace.copy(), exclude_border=False, threshold_rel=threshold, min_distance=window_size)
    peak_values = np.array([hspace[tuple(peaks_indices[j])] for j in range(len(peaks_indices))])
    res = []
    res.append(peak_values)
    for i in range(len(params_list)):
        res.append(params_list[i][peaks_indices.T[i]])
    return res

# GIVEN
def angle_with_x_axis(pi, pj):  
    '''
    Compute the angle that the line connecting two points I and J make with the x-axis (mind our coordinate convention)
    Do note that the line direction is from point I to point J.
    '''
    # get the difference between point p1 and p2
    y, x = pi[0]-pj[0], pi[1]-pj[1] 
    
    if x == 0:
        return np.pi/2  
    
    angle = np.arctan(y/x)
    if angle < 0:
        angle += np.pi
    return angle

# GIVEN
def midpoint(pi, pj):
    '''
    Get y and x coordinates of the midpoint of I and J
    '''
    return (pi[0]+pj[0])/2, (pi[1]+pj[1])/2

# GIVEN
def distance(pi, pj):
    '''
    Compute the Euclidean distance between two points I and J.
    '''
    y,x = pi[0]-pj[0], pi[1]-pj[1] 
    return np.sqrt(x**2+y**2)

# 3.1 IMPLEMENT
def shift_sift_descriptor(desc):
    '''
       Generate a virtual mirror descriptor for a given descriptor.
       Note that you have to shift the bins within a mini histogram, and the mini histograms themselves.
       e.g:
       Descriptor for a keypoint
       (the dimension is (128,), but here we reshape it to (16,8). Each length-8 array is a mini histogram.)
      [[  0.,   0.,   0.,   5.,  41.,   0.,   0.,   0.],
       [ 22.,   2.,   1.,  24., 167.,   0.,   0.,   1.],
       [167.,   3.,   1.,   4.,  29.,   0.,   0.,  12.],
       [ 50.,   0.,   0.,   0.,   0.,   0.,   0.,   4.],
       
       [  0.,   0.,   0.,   4.,  67.,   0.,   0.,   0.],
       [ 35.,   2.,   0.,  25., 167.,   1.,   0.,   1.],
       [167.,   4.,   0.,   4.,  32.,   0.,   0.,   5.],
       [ 65.,   0.,   0.,   0.,   0.,   0.,   0.,   1.],
       
       [  0.,   0.,   0.,   0.,  74.,   1.,   0.,   0.],
       [ 36.,   2.,   0.,   5., 167.,   7.,   0.,   4.],
       [167.,  10.,   0.,   1.,  30.,   1.,   0.,  13.],
       [ 60.,   2.,   0.,   0.,   0.,   0.,   0.,   1.],
       
       [  0.,   0.,   0.,   0.,  54.,   3.,   0.,   0.],
       [ 23.,   6.,   0.,   4., 167.,   9.,   0.,   0.],
       [167.,  40.,   0.,   2.,  30.,   1.,   0.,   0.],
       [ 51.,   8.,   0.,   0.,   0.,   0.,   0.,   0.]]
     ======================================================
       Descriptor for the same keypoint, flipped over the vertical axis
      [[  0.,   0.,   0.,   3.,  54.,   0.,   0.,   0.],
       [ 23.,   0.,   0.,   9., 167.,   4.,   0.,   6.],
       [167.,   0.,   0.,   1.,  30.,   2.,   0.,  40.],
       [ 51.,   0.,   0.,   0.,   0.,   0.,   0.,   8.],
       
       [  0.,   0.,   0.,   1.,  74.,   0.,   0.,   0.],
       [ 36.,   4.,   0.,   7., 167.,   5.,   0.,   2.],
       [167.,  13.,   0.,   1.,  30.,   1.,   0.,  10.],
       [ 60.,   1.,   0.,   0.,   0.,   0.,   0.,   2.],
       
       [  0.,   0.,   0.,   0.,  67.,   4.,   0.,   0.],
       [ 35.,   1.,   0.,   1., 167.,  25.,   0.,   2.],
       [167.,   5.,   0.,   0.,  32.,   4.,   0.,   4.],
       [ 65.,   1.,   0.,   0.,   0.,   0.,   0.,   0.],
       
       [  0.,   0.,   0.,   0.,  41.,   5.,   0.,   0.],
       [ 22.,   1.,   0.,   0., 167.,  24.,   1.,   2.],
       [167.,  12.,   0.,   0.,  29.,   4.,   1.,   3.],
       [ 50.,   4.,   0.,   0.,   0.,   0.,   0.,   0.]]
    '''
    # YOUR CODE HERE
   
    # END
    return res

# 3.1 IMPLEMENT
def create_mirror_descriptors(img):
    '''
    Return the output for compute_cv2_descriptor (which you can find in utils.py)
    Also return the set of virtual mirror descriptors.
    Make sure the virtual descriptors correspond to the original set of descriptors.
    '''
    # YOUR CODE HERE
 
    # END
    return kps, descs, sizes, angles, mir_descs

# 3.2 IMPLEMENT
def match_mirror_descriptors(descs, mirror_descs, threshold = 0.7):
    '''
    First use `top_k_matches` to find the nearest 3 matches for each keypoint. Then eliminate the mirror descriptor that comes 
    from the same keypoint. Perform ratio test on the two matches left. If no descriptor is eliminated, perform the ratio test 
    on the best 2. 
    '''
    three_matches = top_k_matches(descs, mirror_descs, k=3)

    match_result = []
    # YOUR CODE HERE
   
    # END
    return match_result

# 3.3 IMPLEMENT
def find_symmetry_lines(matches, kps):
    '''
    For each pair of matched keypoints, use the keypoint coordinates to compute a candidate symmetry line.
    Assume the points associated with the original descriptor set to be I's, and the points associated with the mirror descriptor set to be
    J's.
    '''
    rhos = []
    thetas = []
    # YOUR CODE HERE

    # END
    
    return rhos, thetas

# 3.4 IMPLEMENT
def hough_vote_mirror(matches, kps, im_shape, window=1, threshold=0.5, num_lines=1):
    '''
    Hough Voting:
                 0<=thetas<= 2pi      , interval size = 1 degree
        -diagonal <= rhos <= diagonal , interval size = 1 pixel
    Feel free to vary the interval size.
    '''
    rhos, thetas = find_symmetry_lines(matches, kps)
    
    # YOUR CODE HERE
  
    # END
    
    return rho_values, theta_values

##################### PART 4 ###################

# 4.1 IMPLEMENT
def match_with_self(descs, kps, threshold=0.8):
    '''
    Use `top_k_matches` to match a set of descriptors against itself and find the best 3 matches for each descriptor.
    Discard the trivial match for each trio (if exists), and perform the ratio test on the two matches left (or best two if no match is removed)
    '''
   
    matches = []
    
    # YOUR CODE HERE
    top_3_matches = top_k_matches(descs, descs, k = 3)
    for i in top_3_matches: 
        if (i[1][1][1] / i[1][2][1])  < threshold:
            matches.append([i[0],i[1][1][0]])
    # END
    # Modify this line as you wish
    matches = np.array(matches)
    # END
    return matches

# 4.2 IMPLEMENT
def find_rotation_centers(matches, kps, angles, sizes, im_shape):
    '''
    For each pair of matched keypoints (using `match_with_self`), compute the coordinates of the center of rotation and vote weight. 
    For each pair (kp1, kp2), use kp1 as point I, and kp2 as point J. The center of rotation is such that if we pivot point I about it,
    the orientation line at point I will end up coinciding with that at point J. 
    
    You may want to draw out a simple case to visualize first.
    
    If a candidate center lies out of bound, ignore it.
    '''
    # Y-coordinates, X-coordinates, and the vote weights 
    Y = []
    X = []
    W = []
    
    # YOUR CODE HERE
    for i in matches:
        kp1 = kps[i[0]]
        kp2 = kps[i[1]]
        angle1 = angles[i[0]]
        angle2 = angles[i[1]]
        if abs(angle1 - angle2) <= 1:
            continue
        else:
            d = distance(kp1, kp2)
            gamma = angle_with_x_axis(kp1, kp2)
            beta = (angle1 - angle2 + np.pi)/2
            r = d * np.sqrt(1 + np.tan(beta)**2) / 2
            x_c = kp1[0] + r * np.cos(beta + gamma)
            y_c = kp1[1] + r * np.sin(beta + gamma)
            if x_c > im_shape[0] or x_c < 0 or y_c > im_shape[1] or y_c < 0:
                pass
            else:
                Y.append(y_c)
                X.append(x_c)
                W.append((sizes[i[0]], sizes[i[1]]))
    # END
    
    return Y,X,W

# 4.3 IMPLEMENT
def hough_vote_rotation(matches, kps, angles, sizes, im_shape, window=1, threshold=0.5, num_centers=1):
    '''
    Hough Voting:
        X: bound by width of image
        Y: bound by height of image
    Return the y-coordianate and x-coordinate values for the centers (limit by the num_centers)
    '''
    
    Y,X,W = find_rotation_centers(matches, kps, angles, sizes, im_shape)
    
    # YOUR CODE HERE
    
    # END
    
    return y_values, x_values
