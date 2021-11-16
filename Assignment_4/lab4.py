import cv2
import numpy as np
import math
from numpy import core
from sklearn.cluster import KMeans

### Part 1

def detect_points(img, min_distance, rou, pt_num, patch_size, tau_rou, gamma_rou):
    """
    Patchwise Shi-Tomasi point extraction.

    Hints:
    (1) You may find the function cv2.goodFeaturesToTrack helpful. The initial default parameter setting is given in the notebook.

    Args:
        img: Input RGB image. 
        min_distance: Minimum possible Euclidean distance between the returned corners. A parameter of cv2.goodFeaturesToTrack
        rou: Parameter characterizing the minimal accepted quality of image corners. A parameter of cv2.goodFeaturesToTrack
        pt_num: Maximum number of corners to return. A parameter of cv2.goodFeaturesToTrack
        patch_size: Size of each patch. The image is divided into several patches of shape (patch_size, patch_size). There are ((h / patch_size) * (w / patch_size)) patches in total given a image of (h x w)
        tau_rou: If rou falls below this threshold, stops keypoint detection for that patch
        gamma_rou: Decay rou by a factor of gamma_rou to detect more points.
    Returns:
        pts: Detected points of shape (N, 2), where N is the number of detected points. Each point is saved as the order of (height-corrdinate, width-corrdinate)
    """
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    h, w, c = img.shape

    Np = pt_num * 0.9 # The required number of keypoints for each patch. `pt_num` is used as a parameter, while `Np` is used as a stopping criterion.

    # YOUR CODE HERE
    P = h // 50
    Q = w // 50
    pts = []
    for i in range(P):
        for j in range(Q):
            Patch = img_gray[i * 50: (i + 1) * 50, j * 50: (j + 1) * 50]
            kp = cv2.goodFeaturesToTrack(Patch, maxCorners=pt_num, qualityLevel=rou, minDistance=min_distance)
            temp_rho = rou
            while (len(kp) <= int(Np)) and (temp_rho > tau_rou):
                temp_rho = temp_rho * gamma_rou
                kp = cv2.goodFeaturesToTrack(Patch, maxCorners=pt_num, qualityLevel=temp_rho, minDistance=min_distance)
            kp = kp.reshape(len(kp), 2)
            for k in range(len(kp)):
                kp[k][0] += i * 50
                kp[k][1] += j * 50
            pts.extend(kp)
    pts = np.array(pts)
    # END

    return pts

def extract_point_features(img, pts, window_patch):
    """
    Extract patch feature for each point.

    The patch feature for a point is defined as the patch extracted with this point as the center.

    Note that the returned pts is a subset of the input pts. 
    We discard some of the points as they are close to the boundary of the image and we cannot extract a full patch.

    Args:
        img: Input RGB image.
        pts: Detected point corners from detect_points().
        window_patch: The window size of patch cropped around the point. The final patch is of size (5 + 1 + 5, 5 + 1 + 5) = (11, 11). The center is the given point.
                      For example, suppose an image is of size (300, 400). The point is located at (50, 60). The window size is 5. 
                      Then, we use the cropped patch, i.e., img[50-5:50+5+1, 60-5:60+5+1], as the feature for that point. The patch size is (11, 11), so the dimension is 11x11=121.
    Returns:
        pts: A subset of the input points. We can extract a full patch for each of these points.
        features: Patch features of the points of the shape (N, (window_patch*2 + 1)^2), where N is the number of points
    """


    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_gray = img_gray.astype(float)
    h, w, c = img.shape

    # YOUR CODE HERE
    new_pts = []
    features = []
    for i in pts:
        x_i = i[0]
        y_i = i[1]
        feature_x_min = int(x_i - window_patch)
        feature_x_max = int(x_i + window_patch + 1)
        feature_y_min = int(y_i - window_patch)
        feature_y_max = int(y_i + window_patch + 1)
        if feature_x_min >= 0 and feature_x_max < h and feature_y_min >= 0 and feature_y_max < w:
            patch = img_gray[feature_x_min:feature_x_max, feature_y_min:feature_y_max]
            new_pts.append(i)
            patch = patch.flatten()
            p_mean = np.mean(patch)
            p_std = np.std(patch)
            patch = (patch - p_mean) / (p_std + 1e-6)
            features.append(patch)
    features = np.array(features)
    pts = np.array(new_pts)
    # End
    return pts, features

def mean_shift_clustering(features, bandwidth, gamma_h = 1.1):
    """
    Mean-Shift Clustering.

    There are various ways of implementing mean-shift clustering. 
    The provided default bandwidth value may not be optimal to your implementation.
    Please fine-tune the bandwidth so that it can give the best result.

    Args:
        img: Input RGB image.
        bandwidth: If the distance between a point and a clustering mean is below bandwidth, this point probably belongs to this cluster.
    Returns:
        clustering: A dictionary, which contains three keys as follows:
                    1. cluster_centers_: a numpy ndarrary of shape [N_c, 2] for the naive point cloud task and [N_c, 121] for the main task (patch features of the point).
                                         Each row is the center of that cluster.
                    2. labels_:  a numpy nadarray of shape [N,], where N is the number of features. 
                                 labels_[i] denotes the label of the i-th feature. The label is between [0, N_c - 1]
                    3. bandwidth: bandwith value
    """
    # YOUR CODE HERE
    prev_centroids = np.copy(features)
    
    temp_h = bandwidth
    while True:
        new_centroids = np.zeros(prev_centroids.shape)
        for i, x in enumerate(prev_centroids):
            neighbours = neighborhood_points(prev_centroids , x, temp_h)
            if len(neighbours) != 0: 
                numerator = 0
                denominator = 0
                distances = np.linalg.norm(neighbours - x, axis = 1)
                weights = gaussian_kernel(distances, bandwidth = temp_h)
                mult = np.dstack(([weights for i in range(neighbours.shape[1])]))
                numerator = np.sum(mult * neighbours, axis = 1)
                denominator = weights.sum()
                new_feature = numerator / denominator
                new_centroids[i] = new_feature[0]
        temp_h = temp_h * gamma_h
        if np.sum(np.abs(prev_centroids - new_centroids)) < 1e-6:
            prev_centroids = new_centroids
            break
        else:
            prev_centroids = new_centroids

  
    prev_centroids = np.around(prev_centroids)
    clusters = np.unique(prev_centroids, axis = 0)
    labels = np.zeros(features.shape[0]).astype(int)
    for i in range(len(prev_centroids)):
        l = np.argwhere((clusters == prev_centroids[i]).all(1))
        labels[i] = int(l[0])
    clustering = {'cluster_centers_': clusters, 'labels_': labels, 'bandwidth': bandwidth}
    # END
    return clustering

def neighborhood_points(features, centroid, bandwidth):
    diff = features - centroid
    norm = np.linalg.norm(diff, axis=1)
    in_bandwidth = features[np.where(norm < bandwidth)]
    return in_bandwidth

def gaussian_kernel(distance, bandwidth = None):
    val = (1/(bandwidth*math.sqrt(2*math.pi))) * np.exp(-0.5*((distance / bandwidth)**2))
    return val

def cluster(img, pts, features, bandwidth, tau1, tau2, gamma_h):
    """
    Group points with similar appearance, then refine the groups.

    "gamma_h" provides another way of fine-tuning bandwidth to avoid the number of clusters becoming too large.
    Alternatively, you can ignore "gamma_h" and fine-tune bandwidth by yourself.

    Args:
        img: Input RGB image.
        pts: Output from `extract_point_features`.
        features: Patch feature of points. Output from `extract_point_features`.
        bandwidth: Window size of the mean-shift clustering. In pdf, the bandwidth is represented as "h", but we use "bandwidth" to avoid the confusion with the image height
        tau1: Discard clusters with less than tau1 points
        tau2: Perform further clustering for clusters with more than tau2 points using K-means
        gamma_h: To avoid the number of clusters becoming too large, tune the bandwidth by gradually increasing the bandwidth by a factor gamma_h
    Returns:
        clusters: A list of clusters. Each cluster is a numpy ndarray of shape [N_cp, 2]. N_cp is the number of points of that cluster.
    """
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_gray = img_gray.astype(float)
    h, w, c = img.shape

    # YOUR CODE HERE
    clustering = mean_shift_clustering(features, bandwidth, gamma_h=gamma_h)
    centers = clustering['cluster_centers_']
    labels = clustering['labels_']
    
    
    points_in_cluster = [(pts[np.where(labels == i)], features[np.where(labels == i)]) for i in range(len(centers))]
    clusters = []
    for cluster in points_in_cluster:
        num_points = len(cluster[0])
        if num_points > tau1:
            if num_points > tau2:
                K = num_points // tau2
                kmeans = KMeans(n_clusters=K).fit(cluster[1])
                i_pts = cluster[0]
                app = [i_pts[np.where(kmeans.labels_ == j)] for j in range(K)]
                clusters.extend(app)
            else:
                clusters.append(cluster[0])
    # END
    return clusters

### Part 2

def transform(point, M):
    input_pts = np.insert(point, 2, values=1, axis=1)
    transformed = M.dot(input_pts.transpose())
    transformed = transformed.transpose().astype(np.float32)
    return transformed

def angle_bw_edges(p0, p1, p2):  
    '''
    Compute the angle that the line connecting two points I and J make with the x-axis (mind our coordinate convention)
    Do note that the line direction is from point I to point J.
    '''
    # get the difference between point p1 and p2
    v1 = p0 - p1
    v2 = p0 - p2
    num = np.dot(v1, v2)
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    
    costheta = num/denom
    angle = np.arccos(costheta)*180/np.pi
    if np.isnan(angle):
        angle = 180
    return angle

def get_proposal(pts_cluster, tau_a, X):
    """
    Get the lattice proposal

    Hints:
    (1) As stated in the lab4.pdf, we give priority to points close to each other when we sample a triplet.
        This statement means that we can start from the three closest points and iterate N_a times.
        There is no need to go through every triplet combination.
        For instance, you can iterate over each point. For each point, you choose 2 of the 10 nearest points. The 3 points form a triplet.
        In this case N_a = num_points * 45.

    (2) It is recommended that you reorder the 3 points. 
        Since {a, b, c} are transformed onto {(0, 0), (1, 0), (0, 1)} respectively, the point a is expected to be the vertex opposite the longest side of the triangle formed by these three points

    (3) Another way of refining the choice of triplet is to keep the triplet whose angle (between the edges <a, b> and <a, c>) is within a certain range.
        The range, for instance, is between 20 degrees and 120 degrees.

    (4) You may find `cv2.getAffineTransform` helpful. However, be careful about the HW and WH ordering when you use this function.

    (5) If two triplets yield the same number of inliers, keep the one with closest 3 points.

    Args:
        pts_cluster: Points within the same cluster.
        tau_a: The threshold of the difference between the transformed corrdinate and integer positions.
               For example, if a point is transformed into (1.1, -2.03), the closest integer position is (1, -2), then the distance is sqrt(0.1^2 + 0.03^2) (for Euclidean distance case).
               If it is smaller than "tau_a", we consider this point as inlier.
        X: When we compute the inliers, we only consider X nearest points to the point "a". 
    Returns:
        proposal: A list of inliers. The first 3 inliers are {a, b, c}. 
                  Each inlier is a dictionary, with key of "pt_int" and "pt" representing the integer positions after affine transformation and original coordinates.
    """
    # YOU CODE HERE
    proposal = []
    
    max_num_inlier = -1
    max_inliers = []
    max_triplet = []
    max_dist = -1
    
    for point in pts_cluster:
        
        #X-nearest points
        X_nearest_ind = np.argsort(np.linalg.norm(pts_cluster - point, axis = 1))[:X + 1]
        X_nearest_pts = pts_cluster[X_nearest_ind]
        
        #RANSAC Loop
        for _ in range(45):       
            #Determine A, B, C
            rand_ind = np.random.randint(1, 10, size = 2)
            indices = np.append(rand_ind, 0)
            three_nearest_pts = X_nearest_pts[indices]
            dists = []
            for i in range(3):
                for j in range(i + 1, 3):
                    dists.append([(i,j), np.linalg.norm(three_nearest_pts[i] - three_nearest_pts[j])])
            dists = np.array(dists)
            max_ind = np.argmax(dists[:,1])
            longest_edge = dists[max_ind][0]
            triplet = []
            if 0 not in longest_edge:
                triplet = [three_nearest_pts[0], three_nearest_pts[longest_edge[0]], three_nearest_pts[longest_edge[1]]]
            elif 1 not in longest_edge:
                triplet = [three_nearest_pts[1], three_nearest_pts[longest_edge[0]], three_nearest_pts[longest_edge[1]]]
            else:
                triplet = [three_nearest_pts[2], three_nearest_pts[longest_edge[0]], three_nearest_pts[longest_edge[1]]]
            
            #Check angle
        
            if 20 < angle_bw_edges(triplet[0], triplet[1], triplet[2]) < 120:
                #Find M
                src = np.array(triplet).astype(np.float32)
                dst = np.array([[0, 0], [1, 0], [0, 1]]).astype(np.float32)
                M = cv2.getAffineTransform(src[:, [1,0]], dst)
                    
                #Transform Coords
                transformed = transform(X_nearest_pts, M)
                transformed_int = np.round(transformed).astype(np.int32)
                unique_ind = np.unique(transformed_int, axis = 0, return_index = True)[1]
                    
                #Remove duplicates
                transformed_int = transformed_int[unique_ind]
                transformed = transformed[unique_ind]
                    
                diff = np.sum((transformed_int - transformed)**2, axis=1)
                        
                num_inliers = np.sum(diff < tau_a)
                #print(f'Inliers : {num_inliers}')
                    
                if num_inliers > max_num_inlier:    
                    max_num_inlier = num_inliers
                    max_inliers = [X_nearest_pts[unique_ind], transformed_int]
                    max_triplet = triplet
                    max_dist = np.sum(np.abs(dists[:,1]))
            
                elif num_inliers == max_num_inlier and np.sum(np.abs(dists[:,1])):
                    max_num_inlier = num_inliers
                    max_inliers = [X_nearest_pts[unique_ind], transformed_int]
                    max_triplet = triplet
                    max_dist = np.sum(np.abs(dists[:,1]))
                else:
                    pass
    
    basis = [[0,0], [1,0], [0,1]]
    for i in range(len(max_triplet)):
        proposal.append({'pt_int': basis[i], 'pt': max_triplet[i]})
    
    if max_inliers != []: 
        for i in range(len(max_inliers[0])):
            if not (max_inliers[1][i].tolist() in basis):
                proposal.append({'pt_int': max_inliers[1][i], 'pt': max_inliers[0][i]})
    # END
    return proposal





def find_texels(img, proposal, texel_size=50):
    """
    Find texels from the given image.

    Hints:
    (1) This function works on RGB image, unlike previous functions such as point detection and clustering that operate on grayscale image.

    (2) You may find `cv2.getPerspectiveTransform` and `cv2.warpPerspective` helpful.
        Please refer to the demo in the notebook for the usage of the 2 functions.
        Be careful about the HW and WH ordering when you use this function.
    
    (3) As stated in the pdf, each texel is defined by 3 or 4 inlier keypoints on the corners.
        If you find this sentence difficult to understand, you can go to check the demo.
        In the demo, a corresponding texel is obtained from 3 points. The 4th point is predicted from the 3 points.


    Args:
        img: Input RGB image
        proposal: Outputs from get_proposal(). Proposal is a list of inliers.
        texel_size: The patch size (U, V) of the patch transformed from the quadrilateral. 
                    In this implementation, U is equal to V. (U = V = texel_size = 50.) The texel is a square.
    Returns:
        texels: A numpy ndarray of the shape (#texels, texel_size, texel_size, #channels).
    """
    # YOUR CODE HERE
    texels = []
    pt_int = []
    pt = []
    for i in proposal:
        pt_int.append(i['pt_int'])
        pt.append(i['pt'])
    
    pt_int = np.array(pt_int)
    pt = np.array(pt)
    
    for i in pt_int:
        dist = np.linalg.norm(pt_int - i, axis = 1)
        dist_valid = np.where(dist <= 1)
        
        selected_pt_int = pt_int[dist_valid]
        selected_pt = pt[dist_valid]
        selected_dist = dist[dist_valid]
        
        sort_ind = np.argsort(selected_dist)[:3]
        sort_pt_int = selected_pt_int[sort_ind]
        sort_pt = selected_pt[sort_ind]
        if sort_ind.shape[0] == 3:
            corners_src = sort_pt.astype(np.float32)
            point_fourth = corners_src[1] + (corners_src[0] - corners_src[1]) + (corners_src[2] - corners_src[1])
            point_fourth = np.expand_dims(point_fourth, axis = 0)
            corners_src = np.append(corners_src, point_fourth, axis = 0)
            corners_dst = np.float32([[ 0,  0],
                          [texel_size,  0],
                          [texel_size, texel_size],
                          [0, texel_size]])

            matrix_projective = cv2.getPerspectiveTransform(corners_src[:, [1, 0]], corners_dst)
            img_demo_warped = cv2.warpPerspective(img, matrix_projective, (texel_size, texel_size))
            texels.append(img_demo_warped)
    texels = np.array(texels)
    # END
    return texels

def score_proposal(texels, a_score_count_min=3):
    """
    Calcualte A-Score.

    Hints:
    (1) Each channel is normalized separately.
        The A-score for a RGB texel is the average of 3 A-scores of each channel.

    (2) You can return 1000 (in our example) to denote a invalid A-score.
        An invalid A-score is usually results from clusters with less than "a_score_count_min" texels.

    Args:
        texels: A numpy ndarray of the shape (#texels, window, window, #channels).
        a_score_count_min: Minimal number of texels we need to calculate the A-score.
    Returns:
        a_score: A-score calculated from the texels. If there are no sufficient texels, return 1000.    
    """

    K, U, V, C = texels.shape

    # YOUR CODE HERE
    
    
    #Normalize Texels
    for i in range(K):
        if C == 3:
            texels[i,:,:,0] = (texels[i,:,:,0] - np.mean(texels[i,:,:,0])) / (np.std(texels[i,:,:,0]) + 1e-10)
            texels[i,:,:,1] = (texels[i,:,:,1] - np.mean(texels[i,:,:,1])) / (np.std(texels[i,:,:,1]) + 1e-10)
            texels[i,:,:,2] = (texels[i,:,:,2] - np.mean(texels[i,:,:,2])) / (np.std(texels[i,:,:,2]) + 1e-10)
        else:
            texels[i,:,:,0] = (texels[i,:,:,0] - np.mean(texels[i,:,:,0])) / (np.std(texels[i,:,:,0]) + 1e-10)
    
    if K < a_score_count_min:
        return 1000        
    else:
        if C == 3:
            R = texels[:,:,:,0]
            G = texels[:,:,:,1]
            B = texels[:,:,:,2]
            
            a_score_r = float(0)
            a_score_g = float(0)
            a_score_b = float(0)
            
            for i in range(U):
                a_score_r += np.std(R[:,i,i])
                a_score_g += np.std(G[:,i,i])
                a_score_b += np.std(B[:,i,i])
            
            a_score_r = a_score_r/(U * V * np.sqrt(K))
            a_score_g = a_score_g/(U * V * np.sqrt(K))
            a_score_b = a_score_b/(U * V * np.sqrt(K))
            
            a_score = (a_score_r + a_score_g + a_score_b ) / 3             
        else:
            a_score = 0
            
            for i in range(U):
                a_score += np.std(texels[:,i,i])
                    
            a_score = a_score/(U * V * np.sqrt(K))
    # END

    return a_score


### Part 3
# You are free to change the input argument of the functions in Part 3.
# GIVEN
def non_max_suppression(response, suppress_range, threshold=None):
    """
    Non-maximum Suppression for translation symmetry detection

    The general approach for non-maximum suppression is as follows:
        1. Perform thresholding on the input response map. Set the points whose values are less than the threshold as 0.
        2. Find the largest response value in the current response map
        3. Set all points in a certain range around this largest point to 0. 
        4. Save the current largest point
        5. Repeat the step from 2 to 4 until all points are set as 0. 
        6. Return the saved points are the local maximum.

    Args:
        response: numpy.ndarray, output from the normalized cross correlation
        suppress_range: a tuple of two ints (H_range, W_range). The points around the local maximum point within this range are set as 0. In this case, there are 2*H_range*2*W_range points including the local maxima are set to 0
    Returns:
        threshold: int, points with value less than the threshold are set to 0
    """
    H, W = response.shape[:2]
    H_range, W_range = suppress_range
    res = np.copy(response)

    if threshold is not None:
        res[res<threshold] = 0

    idx_max = res.reshape(-1).argmax()
    x, y = idx_max // W, idx_max % W
    point_set = set()
    while res[x, y] != 0:
        point_set.add((x, y))
        res[max(x - H_range, 0): min(x+H_range, H), max(y - W_range, 0):min(y+W_range, W)] = 0
        idx_max = res.reshape(-1).argmax()
        x, y = idx_max // W, idx_max % W
    for x, y in point_set:
        res[x, y] = response[x, y]
    return res

def template_match(img, proposal, threshold):
    """
    Perform template matching on the original input image.

    Hints:
    (1) You may find cv2.copyMakeBorder and cv2.matchTemplate helpful. The cv2.copyMakeBorder is used for padding.
        Alternatively, you can use your implementation in Lab 1 for template matching.

    (2) For non-maximum suppression, you can either use the one you implemented for lab 1 or the code given above.

    Returns:
        response: A sparse response map from non-maximum suppression. 
    """
    # YOUR CODE HERE
    # proposal format = list of inliers {'pt_int': array([0, 0]), 'pt': array([ 51, 694])}. 'pt' is actual inlier
    # Take top 3 inliers (int(proposal[0]['pt'][1]), int(proposal[0]['pt'][0])), (int(proposal[1]['pt'][1]), int(proposal[1]['pt'][0])), (int(proposal[2]['pt'][1]), int(proposal[2]['pt'][0]))
    a = [proposal[0]['pt'][1], proposal[0]['pt'][0]]
    b = [proposal[1]['pt'][1], proposal[1]['pt'][0]]
    c = [proposal[2]['pt'][1], proposal[2]['pt'][0]]
    # midpointBC = ((b[0] + c[0]) / 2, (b[1] + c[1]) / 2)
    # distAToMidBC = (midpointBC[0] - a[0], midpointBC[1] - a[1])
    # d = (midpointBC[0] + distAToMidBC, midpointBC[1] + distAToMidBC)

    d = a + (b - a) + (c - a)

    # Template left top edge = (b_x, d_y), right bottom edge = (c_x, a_y)
    template = img[d[1]:a[1], b[0]:c[0], :]
    top_bottom_pad = int(template.shape[0] / 2)
    left_right_pad = int(template.shape[1] / 2)
    img_padded = cv2.copyMakeBorder(img, top_bottom_pad, top_bottom_pad, left_right_pad, left_right_pad, cv2.BORDER_CONSTANT, None, 0)
    response = cv2.matchTemplate(img_padded, template, cv2.TM_CCOEFF_NORMED)
    response = non_max_suppression(response, template.shape, threshold)

    # END
    return response

def maxima2grid(img, proposal, response):
    """
    Estimate 4 lattice points from each local maxima.

    Hints:
    (1) We can transfer the 4 offsets between the center of the original template and 4 lattice unit points to new detected centers.

    Args:
        response: The response map from `template_match()`.

    Returns:
        points_grid: an numpy ndarray of shape (N, 2), where N is the number of grid points.
    
    """
    # YOUR CODE HERE
    a = [proposal[0]['pt'][1], proposal[0]['pt'][0]]
    b = [proposal[1]['pt'][1], proposal[1]['pt'][0]]
    c = [proposal[2]['pt'][1], proposal[2]['pt'][0]]
    d = a + (b - a) + (c - a)
    offset1 = (d - a) / 2
    offset2 = (c - b) / 2

    points_grid = []
    maxima = []
    for i, j in response.shape:
        if response[i, j] > 0:
            maxima.append([i, j])
            center = np.array([i, j])
            points_grid.append(center - offset1)
            points_grid.append(center + offset1)
            points_grid.append(center - offset2)
            points_grid.append(center + offset2)
    points_grid = np.array(points_grid)
    maxima = np.array(maxima)
    # END

    return points_grid, maxima

def refine_grid(img, proposal, points_grid, maxima):
    """
    Refine the detected grid points.

    Args:
        points_grid: The output from the `maxima2grid()`.

    Returns:
        points: A numpy ndarray of shape (N, 2), where N is the number of refined grid points.
    """
    # YOUR CODE HERE
    a = [proposal[0]['pt'][1], proposal[0]['pt'][0]]
    b = [proposal[1]['pt'][1], proposal[1]['pt'][0]]
    c = [proposal[2]['pt'][1], proposal[2]['pt'][0]]
    d = a + (b - a) + (c - a)
    points = []

    merge_points = []
    # points = np.unique(points_grid, axis=0)
    for i in range(0, points_grid.shape[0]-1):
        for j in range(i+1, points_grid.shape):
            if points_grid[i] == points_grid[j]:
                merge_points.append([i, j])
                break
        points.append(points_grid[i])
    
    for merge_point in merge_points:
        # Every 4 grid points produced from one maximum
        maxi1 = maxima[merge_point[0] / 4]
        maxi2 = maxima[merge_point[1] / 4]
        new_point = (maxi1 + maxi2) / 2
        points.append(new_point)

    points = np.array(points)
    # END

    return points

def grid2latticeunit(img, proposal, points):
    """
    Convert each lattice grid point into integer lattice grid.

    Hints:
    (1) Since it is difficult to know whether two points should be connected, one way is to map each point into an integer position.
        The integer position should maintain the spatial relationship of these points.
        For instance, if we have three points x1=(50, 50), x2=(70, 50) and x3=(70, 70), we can map them (4, 5), (5, 5) and (5, 6).
        As the distances between (4, 5) and (5, 5), (5, 5) and (5, 6) are both 1, we know that (x1, x2) and (x2, x3) form two edges.
    
    (2) You can use affine transformation to build the mapping above, but do not perform global affine transformation.

    (3) The mapping in the hints above are merely to know whether two points should be connected. 
        If you have your own method for finding the relationship, feel free to implement your owns and ignore the hints above.


    Returns:
        edges: A list of edges in the lattice structure. Each edge is defined by two points. The point coordinate is in the image coordinate.
    """

    # # YOUR CODE HERE
    a = [proposal[0]['pt'][1], proposal[0]['pt'][0]]
    b = [proposal[1]['pt'][1], proposal[1]['pt'][0]]
    c = [proposal[2]['pt'][1], proposal[2]['pt'][0]]
    
    ori_grid = np.array([a, b, c]).astype(np.float32)
    final_grid = np.array([[0, 0], [1, 0], [0, 1]]).astype(np.float32)
    M = cv2.getAffineTransform(ori_grid, final_grid)

    edges = []
    # Transform each pair of grid points & check their distance
    for i in range(points.shape[0]-1):
        for j in range(i+1, points.shape[0]):
            affine_i = transform(points[i])
            affine_j = transform(points[j])
            if int(np.linalg.norm(affine_i - affine_j)) == 1:
                edges.append([points[i], points[j]])

    return edges

### Part 4

def detect_points_whole(img, min_distance, rou, N_p, patch_size, tau_rou, gamma_rou):
    """
    Patchwise Shi-Tomasi point extraction.

    Hints:
    (1) You may find the function cv2.goodFeaturesToTrack helpful. The initial default parameter setting is given in the notebook.

    Args:
        img: Input RGB image. 
        min_distance: Minimum possible Euclidean distance between the returned corners. A parameter of cv2.goodFeaturesToTrack
        rou: Parameter characterizing the minimal accepted quality of image corners. A parameter of cv2.goodFeaturesToTrack
        pt_num: Maximum number of corners to return. A parameter of cv2.goodFeaturesToTrack
        patch_size: Size of each patch. The image is divided into several patches of shape (patch_size, patch_size). There are ((h / patch_size) * (w / patch_size)) patches in total given a image of (h x w)
        tau_rou: If rou falls below this threshold, stops keypoint detection for that patch
        gamma_rou: Decay rou by a factor of gamma_rou to detect more points.
    Returns:
        pts: Detected points of shape (N, 2), where N is the number of detected points. Each point is saved as the order of (height-corrdinate, width-corrdinate)
    """
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    h, w, c = img.shape

    # YOUR CODE HERE
    pts = []
    P = h // 50
    Q = w // 50
    max_corners = P * Q * N_p
    kp = cv2.goodFeaturesToTrack(img_gray, maxCorners=max_corners, qualityLevel=rou, minDistance=min_distance)
    temp_rho = rou
    while (len(kp) <= int(max_corners)) and (temp_rho > tau_rou):
        temp_rho = temp_rho * gamma_rou
        kp = cv2.goodFeaturesToTrack(img_gray, maxCorners=max_corners, qualityLevel=temp_rho, minDistance=min_distance)
    kp = kp.reshape(len(kp), 2)
    pts.extend(kp)
    pts = np.array(pts)
    # END

    return pts



