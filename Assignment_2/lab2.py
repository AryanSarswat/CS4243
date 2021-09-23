from skimage.draw import circle_perimeter
from skimage.feature import peak_local_max
import math
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import patches


##################### TASK 1 ###################

# 1.1 IMPLEMENT
def make_gaussian_kernel(ksize, sigma):
    '''
    Implement the simplified Gaussian kernel below:
    k(x,y)=exp(((x-x_mean)^2+(y-y_mean)^2)/(-2sigma^2))
    Make Gaussian kernel be central symmentry by moving the 
    origin point of the coordinate system from the top-left
    to the center. Please round down the mean value. In this assignment,
    we define the center point (cp) of even-size kernel to be the same as that of the nearest
    (larger) odd size kernel, e.g., cp(4) to be same with cp(5).
    :param ksize: int
    :param sigma: float
    :return kernel: numpy.ndarray of shape (ksize, ksize)
    '''

    # YOUR CODE HERE
    if ksize % 2 == 0:
        x_mean = y_mean = (ksize // 2) + 1
    else:
        x_mean = y_mean = ksize // 2

    def k(x, y): return np.exp(((x-x_mean)**2 + (y-y_mean)**2)/(-2*(sigma**2)))
    kernel = np.zeros((ksize, ksize))
    for x in range(ksize):
        for y in range(ksize):
            kernel[x][y] = k(x, y)
    # END

    return kernel / kernel.sum()

# GIVEN


def cs4243_filter(image, kernel):
    """
    Fast version of filtering algorithm.
    Pre-extract all the regions of kernel size,
    and obtain a matrix of shape (Hi*Wi, Hk*Wk), also reshape the flipped
    kernel to be of shape (Hk*Wk, 1), then do matrix multiplication, and reshape back
    to get the final output image. 
    :param image: numpy.ndarray
    :param kernel: numpy.ndarray
    :return filtered_image: numpy.ndarray
    """
    def cs4243_rotate180(kernel):
        kernel = np.flip(np.flip(kernel, 0), 1)
        return kernel

    def img2col(input, h_out, w_out, h_k, w_k, stride):
        h, w = input.shape
        out = np.zeros((h_out*w_out, h_k*w_k))

        convwIdx = 0
        convhIdx = 0
        for k in range(h_out*w_out):
            if convwIdx + w_k > w:
                convwIdx = 0
                convhIdx += stride
            out[k] = input[convhIdx:convhIdx+h_k,
                           convwIdx:convwIdx+w_k].flatten()
            convwIdx += stride
        return out

    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    if Hk % 2 == 0 or Wk % 2 == 0:
        raise ValueError

    hkmid = Hk//2
    wkmid = Wk//2

    image = cv2.copyMakeBorder(
        image, hkmid, hkmid, wkmid, wkmid, cv2.BORDER_REFLECT)
    filtered_image = np.zeros((Hi, Wi))
    kernel = cs4243_rotate180(kernel)
    col = img2col(image, Hi, Wi, Hk, Wk, 1)
    kernel_flatten = kernel.reshape(Hk*Wk, 1)
    output = col @ kernel_flatten
    filtered_image = output.reshape(Hi, Wi)

    return filtered_image

# GIVEN


def cs4243_blur(img, gaussian_kernel, display=True):
    '''
    Performing Gaussian blurring on an image using a Gaussian kernel.
    :param img: input image
    :param gaussian_kernel: gaussian kernel
    :return blurred_img: blurred image
    '''

    blurred_img = cs4243_filter(img, gaussian_kernel)

    if display:
        fig1, axes_array = plt.subplots(1, 2)
        fig1.set_size_inches(8, 4)
        image_plot = axes_array[0].imshow(img, cmap=plt.cm.gray)
        axes_array[0].axis('off')
        axes_array[0].set(title='Original Image')
        image_plot = axes_array[1].imshow(blurred_img, cmap=plt.cm.gray)
        axes_array[1].axis('off')
        axes_array[1].set(title='Filtered Image')
        plt.show()
    return blurred_img

# 2 IMPLEMENT


def estimate_gradients(original_img, display=True):
    '''
    Compute gradient orientation and magnitude for the input image.
    Perform the following steps:

    1. Compute dx and dy, responses to the vertical and horizontal Sobel kernel. Make use of the cs4243_filter function.

    2. Compute the gradient magnitude which is equal to sqrt(dx^2 + dy^2) 

    3. Compute the gradient orientation using the following formula:
        gradient = atan2(dy/dx)

    You may want to divide the original image pixel value by 255 to prevent overflow.

    Note that our axis choice is as follows:
            --> y
            |   
            ↓ x    
    Where img[x,y] denotes point on the image at coordinate (x,y)

    :param original_img: original grayscale image
    :return d_mag: gradient magnitudes matrix
    :return d_angle: gradient orientation matrix (in radian)
    '''

    dx = None
    dy = None
    d_mag = None
    d_angle = None

    # YOUR CODE HERE
    '''
    HINT:
    In the lecture, 
    
    Sx =  1  0 -1
          2  0 -2
          1  0 -1
          
    Sy =  1  2  1
          0  0  0
         -1 -2 -1
         
    Here:
    
    Kx = [[ 1,  2,  1],
          [ 0,  0,  0],
          [-1, -2, -1]]
    
    Ky = [[ 1,  0, -1],
          [ 2,  0, -2],
          [ 1,  0, -1]]
 
    This is because x direction is the downward line.
    '''
    Kx = np.array(
        [[-1,  -2, -1],
         [0,  0,  0],
         [1, 2, 1]]
    )

    Ky = np.array(
        [[-1,  0, 1],
         [-2,  0, 2],
         [-1,  0, 1]]
    )

    normalized = original_img / 255.0
    dx = cs4243_filter(normalized, Kx)
    dy = cs4243_filter(normalized, Ky)

    d_mag = np.sqrt((dx**2) + (dy**2))
    d_mag = (d_mag/np.linalg.norm(d_mag))*255
    d_angle = np.arctan2(dy,dx)

    # END
    if display:

        fig2, axes_array = plt.subplots(1, 4)
        fig2.set_size_inches(16, 4)
        image_plot = axes_array[0].imshow(d_mag, cmap='gray')
        axes_array[0].axis('off')
        axes_array[0].set(title='Gradient Magnitude')

        image_plot = axes_array[1].imshow(dx, cmap='gray')
        axes_array[1].axis('off')
        axes_array[1].set(title='dX')

        image_plot = axes_array[2].imshow(dy, cmap='gray')
        axes_array[2].axis('off')
        axes_array[2].set(title='dY')

        image_plot = axes_array[3].imshow(d_angle, cmap='gray')
        axes_array[3].axis('off')
        axes_array[3].set(title='Gradient Direction')
        plt.show()

    return d_mag, d_angle

# 3a IMPLEMENT


def non_maximum_suppression(d_mag, d_angle, display=True):
    '''
    Perform non-maximum suppression on the gradient magnitude matrix without interpolation.
    Split the range -180° ~ 180° into 8 even ranges. For each pixel, determine which range the gradient
    orientation belongs to and pick the corresponding two pixels from the adjacent eight pixels surrounding 
    that pixel. Keep the pixel if its value is larger than the other two.
    Do note that the coordinate system is as below and angular measurements are clockwise.
    ----------→ y  
    |
    |
    |
    |        x X x
    ↓ x       \|/   
             x-o-x  
              /|\    
             x X x 
         -22.5 0 22.5

    For instance, in the example above if the orientation at the coordinate of interest (x,y) is 20°, it belongs to the -22.5°~22.5° range, and the two pixels to be compared with are at (x+1,y) and (x-1,y) (aka the two big X's). If the angle was instead 40°, it belongs to the 22.5°-67.5° and the two pixels we need to consider will be (x+1, y+1) and (x-1,y-1)

    There are only 4 sets of offsets: (0,1), (1,0), (1,1), and (1,-1), since to find the second pixel offset you just need 
    to multiply the first tuple by -1.

    :param d_mag: gradient magnitudes matrix
    :param d_angle: gradient orientation matrix (in radian)
    :return out: non-maximum suppressed image
    '''

    out = np.zeros(d_mag.shape, d_mag.dtype)
    # Change angles to degrees to improve quality of life
    d_angle_180 = d_angle * 180/np.pi

    # YOUR CODE HERE
    x_range, y_range = out.shape
    for x in range(1, x_range - 1):
        for y in range(1, y_range - 1):
            angle = d_angle_180[x][y]
            check = d_mag[x][y]
            if -22.5 <= angle < 22.5 or 157.5 <= angle <= 180 or -180 <= angle < -157.5:
                check1 = d_mag[x + 1][y]
                check2 = d_mag[x - 1][y]
                if check > check1 and check > check2:
                    out[x][y] = check
            elif -67.5 <= angle < -22.5 or 112.5 <= angle < 157.5:
                check1 = d_mag[x + 1][y - 1]
                check2 = d_mag[x - 1][y + 1]
                if check > check1 and check > check2:
                    out[x][y] = check
            elif -112.5 <= angle < -67.5 or 67.5 <= angle < 112.5:
                check1 = d_mag[x][y + 1]
                check2 = d_mag[x][y - 1]
                if check > check1 and check > check2:
                    out[x][y] = check
            else: # -157.5 <= angle < -112.5 or 22.5 <= angle < 67.5
                check1 = d_mag[x + 1][y + 1]
                check2 = d_mag[x - 1][y - 1]
                if check > check1 and check > check2:
                    out[x][y] = check

    # END
    if display:
        _ = plt.figure(figsize=(10, 10))
        plt.imshow(out)
        plt.title("Suppressed image (without interpolation)")

    return out

# 3b IMPLEMENT


def non_maximum_suppression_interpol(d_mag, d_angle, display=True):
    '''
    Perform non-maximum suppression on the gradient magnitude matrix with interpolation.
    :param d_mag: gradient magnitudes matrix
    :param d_angle: gradient orientation matrix (in radian)
    :return out: non-maximum suppressed image
    '''

    out = np.zeros(d_mag.shape, d_mag.dtype)
    d_angle_180 = d_angle * 180/np.pi

    # YOUR CODE HERE
    x_range, y_range = out.shape
    for x in range(1, x_range - 1):
        for y in range(1, y_range - 1):
            angle = d_angle_180[x][y]
            check = d_mag[x][y]
            if 0 <= angle <= 45 or -180 <= angle < -135:
                factor = np.tan((angle/180) * np.pi) if angle > 0 else - np.tan((angle/180) * np.pi)
                check1 = factor * (d_mag[x + 1][y + 1] - d_mag[x + 1][y]) + d_mag[x + 1][y]
                check2 = factor * (d_mag[x - 1][y - 1] - d_mag[x - 1][y]) + d_mag[x - 1][y]
                if check > check1 and check > check2:
                    out[x][y] = check
            elif 45 < angle <= 90 or -135 <= angle < -90:
                angle = 90 - angle if angle > 0 else 90 + angle
                factor = np.tan((angle/180) * np.pi) if angle > 0 else - np.tan((angle/180) * np.pi)
                check1 = factor * (d_mag[x + 1][y + 1] - d_mag[x][y + 1]) + d_mag[x][y + 1]
                check2 = factor * (d_mag[x - 1][y - 1] - d_mag[x][y - 1]) + d_mag[x][y - 1]
                if check > check1 and check > check2:
                    out[x][y] = check
            elif 90 < angle <= 135 or -90 <= angle < -45:
                angle = angle - 90 if angle > 0 else 90 + angle
                factor = np.tan((angle/180) * np.pi) if angle > 0 else - np.tan((angle/180) * np.pi)
                check1 = factor * (d_mag[x + 1][y - 1] - d_mag[x][y + 1]) + d_mag[x][y + 1]
                check2 = factor * (d_mag[x - 1][y + 1] - d_mag[x][y - 1]) + d_mag[x][y - 1]
                if check > check1 and check > check2:
                    out[x][y] = check
            elif 135 < angle <= 180 or -45 <= angle < 0:
                angle = 180 - angle if angle > 0 else angle
                factor = np.tan((angle/180) * np.pi) if angle > 0 else - np.tan((angle/180) * np.pi)
                check1 = factor * (d_mag[x - 1][y + 1] - d_mag[x - 1][y]) + d_mag[x - 1][y]
                check2 = factor * (d_mag[x + 1][y - 1] - d_mag[x + 1][y]) + d_mag[x + 1][y]
                if check > check1 and check > check2:
                    out[x][y] = check
    # END
    if display:
        _ = plt.figure(figsize=(10, 10))
        plt.imshow(out, cmap='gray')
        plt.title("Suppressed image (with interpolation)")

    return out

# 4 IMPLEMENT


def double_thresholding(inp, perc_weak=0.1, perc_strong=0.3, display=True):
    '''
    Perform double thresholding. Use on the output of NMS. The high and low thresholds are computed as follow:

    delta = max_val - min_val
    high_threshold = min_val + perc_strong * delta 
    low_threshold = min_val + perc_weak * delta

    perc_weak being 0 is possible
    Do note that the return edge images should be binary (0-1 or True-False)
    :param inp: numpy.ndarray
    :param perc_weak: value to determine low threshold
    :param perc_strong: value to determine high threshold
    :return weak_edges, strong_edges: binary edge images
    '''
    weak_edges = strong_edges = None

    # YOUR CODE HERE
    min_val, max_val = np.min(inp), np.max(inp)
    delta = max_val - min_val
    high_threshold = min_val + perc_strong * delta 
    low_threshold = min_val + perc_weak * delta

    strong_edges = np.where(inp > high_threshold, 1, 0)
    weak_edges = np.where(low_threshold < inp, 1, 0)
    weak_edges = weak_edges - strong_edges
    weak_edges = np.where(weak_edges >= 0, weak_edges, 0)
    # END

    if display:

        fig2, axes_array = plt.subplots(1, 2)
        fig2.set_size_inches(10, 5)
        image_plot = axes_array[0].imshow(strong_edges, cmap='gray')
        axes_array[0].axis('off')
        axes_array[0].set(title='Strong ')

        image_plot = axes_array[1].imshow(weak_edges, cmap='gray')
        axes_array[1].axis('off')
        axes_array[1].set(title='Weak')

    return weak_edges, strong_edges

# 5 IMPLEMENT


def edge_linking(weak, strong, n=200, display=True):
    '''
    Perform edge-linking on two binary weak and strong edge images. 
    A weak edge pixel is linked if any of its eight surrounding pixels is a strong edge pixel.
    You may want to avoid using loops directly due to the high computational cost. One possible trick is to generate
    8 2D arrays from the strong edge image by offseting and sum them together; entries larger than 0 mean that at least one surrounding
    pixel is a strong edge pixel (otherwise the sum would be 0).

    You may also want to limit the number of iterations (test with 10-20 iterations first to check your implementation speed), and use a stopping condition (stop if no more pixel is added to the strong edge image).
    Also, when a weak edge pixel is added to the strong set, remember to remove it.


    :param weak: weak edge image (binary)
    :param strong: strong edge image (binary)
    :param n: maximum number of iterations
    :return out: final edge image
    '''
    assert weak.shape == strong.shape, "Weak and strong edge image have to have the same dimension"
    out = None

    # YOUR CODE HERE
    x_limit, y_limit = strong.shape
    for iteration in range(n):
        added = False

        x_ind, y_ind = np.where(strong == 1)

        for ind in zip(x_ind, y_ind):
            neighbors = [(ind[0], ind[1] + 1), (ind[0] + 1, ind[1]), (ind[0] + 1, ind[1] + 1), (ind[0], ind[1] - 1), (ind[0] - 1, ind[1]), (ind[0] - 1, ind[1] - 1) ,
            (ind[0] - 1, ind[1] + 1), (ind[0] + 1, ind[1] - 1)]

            for neighbor_ind in neighbors:
                if 0 <= neighbor_ind[0] < x_limit and 0 <= neighbor_ind[1] < y_limit:
                    if weak[neighbor_ind[0]][neighbor_ind[1]] == 1:
                        added = True
                        strong[neighbor_ind[0]][neighbor_ind[1]] = 1
                        weak[neighbor_ind[0]][neighbor_ind[1]] = 0
        if not added:
            break
    out = strong
    # END
    if display:
        _ = plt.figure(figsize=(10, 10))
        plt.imshow(out)
        plt.title("Edge image")
    return out

##################### TASK 2 ######################

# 1/2/3 IMPLEMENT


def hough_vote_lines(img):
    '''
    Use the edge image to vote for 2 parameters: distance and theta
    Beware of our coordinate convention.

    You may find the np.linspace function useful.

    :param img: edge image
    :return A: accumulator array
    :return distances: distance values array
    :return thetas: theta values array
    '''
    # YOUR CODE HERE
    dist_interval = 1
    theta_interval = math.pi/180
    img_height, img_width = img.shape
    dist_max = math.sqrt(img_height**2 + img_width**2)
    dist_min = -dist_max
    theta_max = math.pi
    theta_min = 0

    # Quantization
    dist_bin_num = math.ceil((dist_max - dist_min) / dist_interval)
    theta_bin_num = math.ceil((theta_max - theta_min) / theta_interval)
    A = np.zeros((dist_bin_num, theta_bin_num), int)
    # Use arange instead of linspace as dividing bins evenly may result in wrong interval if desired interval not a factor of range size
    distances = np.arange(dist_min, dist_max, dist_interval)
    thetas = np.arange(theta_min, theta_max, theta_interval)
    
    # Voting
    for x in range(img_height):
        for y in range(img_width):
            if img[x][y] <= 0:
                continue
            for i in range(len(thetas)):
                t = thetas[i]
                dist = x * np.cos(t) + y * np.sin(t)
                # Find 1st True value as 1st occurrence of max, bin is <= input
                j = np.argmax(distances > dist) - 1
                A[j, i] += 1
                # print(x, y, t, dist, distances[j])
    
    # END

    return A, distances, thetas


# 4 GIVEN


def find_peak_params(hspace, params_list,  window_size=1, threshold=0.5):
    '''
    Given a Hough space and a list of parameters range, compute the local peaks
    aka bins whose count is larger max_bin * threshold. The local peaks are computed
    over a space of size (2*window_size+1)^(number of parameters).

    Also include the array of values corresponding to the bins, in descending order.

    e.g.
    Suppose for a line detection case, you get the following output:
    [
    [122, 101, 93],
    [3,   40,  21],
    [0,   1.603, 1.605]
    ]
    This means that the local maxima with the highest vote gets a vote score of 122, and the corresponding parameter value is distance=3, 
    theta = 0.
    '''
    assert len(hspace.shape) == len(params_list), "The Hough space dimension does not match the number of parameters"
    for i in range(len(params_list)):
        assert hspace.shape[i] == len(params_list[i]), f"Parameter length does not match size of the corresponding dimension:{len(params_list[i])} vs {hspace.shape[i]}"
    peaks_indices = peak_local_max(hspace.copy(
    ), exclude_border=False, threshold_rel=threshold, min_distance=window_size)
    peak_values = np.array([hspace[tuple(peaks_indices[j])]
                           for j in range(len(peaks_indices))])
    res = []
    res.append(peak_values)
    for i in range(len(params_list)):
        res.append(params_list[i][peaks_indices.T[i]])
    return res


##################### TASK 3 ######################

# 1/2/3 IMPLEMENT


def hough_vote_circles(img, radius=None):
    '''
    Use the edge image to vote for 3 parameters: circle radius and circle center coordinates.
    We also accept a range of radii to save computation costs. If the radius range is not given, it is default to
    [3, diagonal of the circle]. This parameter is very useful for your experimentation later on (e.g. if there are only large circles then you don't have to keep R_min very small).

    Hint: You can use the function circle_perimeter to make a circular mask. Center the mask over the accumulator array and increment the array. In this case, you will have to pad the accumulator array first, and clip it afterwards. 
    Remember that the return accumulator array should have matching dimension with the lengths of the parameter ranges. 

    The dimensions of the accumulator array should be in this order: radius, x-coordinate, y-coordinate.

    :param img: edge image
    :param radius: min radius, max radius
    :return A: accumulator array
    :return R: radius values array
    :return X: x-coordinate values array
    :return Y: y-coordinate values array
    '''

    # Check the radius range
    h, w = img.shape[:2]
    if radius == None:
        R_max = np.hypot(h, w)
        R_min = 3
    else:
        [R_min, R_max] = radius

    # YOUR CODE HERE
    R_interval = 1
    a_interval = 1
    b_interval = 1
    img_height, img_width = img.shape
    a_max = img_height
    b_max = img_width

    # Quantization
    R_bin_num = math.ceil((R_max - R_min) / R_interval)
    a_bin_num = math.ceil(a_max / a_interval)
    b_bin_num = math.ceil(b_max / b_interval)
    A = np.zeros((R_bin_num, a_bin_num, b_bin_num))
    # Use arange instead of linspace as dividing bins evenly may result in wrong interval if desired interval not a factor of range size
    R = np.arange(R_min, R_max, R_interval)
    X = np.arange(0, a_max, a_interval)
    Y = np.arange(0, b_max, b_interval)

    # Weights for circles of different radii, weight decreases as radii increases
    R_weighted_incr = np.linspace(1, 1 + ((R_max - R_min) / R_min), R_bin_num)[::-1]
    
    # Voting
    for x in range(img_height):
        for y in range(img_width):
            if img[x][y] <= 0:
                continue

            for i in range(len(R)):
                r = R[i]
                perim_x, perim_y = circle_perimeter(x, y, r, shape=(img.shape)) # From imported skimage.draw module
                if R_interval == 1 and a_interval == 1 and b_interval == 1:
                    A[i, perim_x, perim_y] += R_weighted_incr[i]

                # Finding bins if intervals != 1 inefficient due to loop over features
                else:
                    for n in range(len(perim_x)):
                        # Find 1st True value as 1st occurrence of max, bin is <= input
                        j = np.argmax(X > perim_x[n]) - 1
                        k = np.argmax(Y > perim_y[n]) - 1
                        if j < 0 or k < 0:
                            continue
                        A[i, j, k] += R_weighted_incr[i]

    # END

    return A, R, X, Y

##################### TASK 4 ######################

# IMPLEMENT


def hough_vote_circles_grad(img, d_angle, radius=None):
    '''
    Use the edge image to vote for 3 parameters: circle radius and circle center coordinates.
    We also accept a range of radii to save computation costs. If the radius range is not given, it is default to
    [3, diagonal of the circle].
    This time, gradient information is used to avoid casting too many unnecessary votes.

    Remember that for a given pixel, you need to cast two votes along the orientation line. One in the positive direction, the other in
    negative direction.

    :param img: edge image
    :param d_angle: corresponding gradient orientation matrix
    :param radius: min radius, max radius
    :return A: accumulator array
    :return R: radius values array
    :return X: x-coordinate values array
    :return Y: y-coordinate values array
    '''
    # Check the radius range
    h, w = img.shape[:2]
    if radius == None:
        R_max = np.hypot(h, w)
        R_min = 3
    else:
        [R_min, R_max] = radius

    # YOUR CODE HERE
    R_interval = 5
    a_interval = 5
    b_interval = 5
    img_height, img_width = img.shape
    a_max = img_height
    b_max = img_width

    # Quantization
    R_bin_num = math.ceil((R_max - R_min) / R_interval)
    a_bin_num = math.ceil(a_max / a_interval)
    b_bin_num = math.ceil(b_max / b_interval)
    A = np.zeros((R_bin_num, a_bin_num, b_bin_num))
    # Use arange instead of linspace as dividing bins evenly may result in wrong interval if desired interval not a factor of range size
    R = np.arange(R_min, R_max, R_interval)
    X = np.arange(0, a_max, a_interval)
    Y = np.arange(0, b_max, b_interval)

    # Weights for circles of different radii, weight decreases as radii increases
    R_weighted_incr = np.ones((R_bin_num))
    
    # Voting
    for x in range(img_height):
        for y in range(img_width):
            if img[x][y] <= 0:
                continue

            for i in range(len(R)):
                r = R[i]
                theta = d_angle[x, y]
                a1 = round(x + r * np.cos(theta))
                b1 = round(y + r * np.sin(theta))
                a2 = round(x - r * np.cos(theta))
                b2 = round(x - r * np.sin(theta))
                
                if R_interval == 1 and a_interval == 1 and b_interval == 1:
                    if (a1 >= 0 and a1 < img_height) and (b1 >= 0 and b1 < img_height):
                        A[i, a1, b1] += R_weighted_incr[i]
                    if (a2 >= 0 and a2 < img_height) and (b2 >= 0 and b2 < img_height):
                        A[i, a2, b2] += R_weighted_incr[i]

                # Place into bins for intervals > 1
                else:
                    # Find 1st True value as 1st occurrence of max, bin is <= input
                    j1 = np.argmax(X > a1) - 1
                    k1 = np.argmax(Y > b1) - 1
                    j2 = np.argmax(X > a2) - 1
                    k2 = np.argmax(Y > b2) - 1
                    if (j1 >= 0 and j1 < img_height) and (k1 >= 0 and k1 < img_height):
                        A[i, j1, k1] += R_weighted_incr[i]
                    if (j2 >= 0 and j2 < img_height) and (k2 >= 0 and k2 < img_height):
                        A[i, j2, k2] += R_weighted_incr[i]

    # END
    return A, R, X, Y


###############################################
"""Helper functions: You should not have to touch the following functions.
"""


def read_img(filename):
    '''
    Read HxWxC image from the given filename
    :return img: numpy.ndarray, size (H, W, C) for RGB. The value is between [0, 255].
    '''
    img = cv2.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def draw_lines(hspace, dists, thetas, hs_maxima, file_path):
    im_c = read_img(file_path)
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    ax = axes.ravel()

    ax[0].imshow(im_c, cmap=cm.gray)
    ax[0].set_title('Input image')
    ax[0].set_axis_off()

    angle_step = 0.5 * np.diff(thetas).mean()
    d_step = 0.5 * np.diff(dists).mean()
    bounds = [np.rad2deg(thetas[0] - angle_step),
              np.rad2deg(thetas[-1] + angle_step),
              dists[-1] + d_step, dists[0] - d_step]

    ax[1].imshow(np.log(1 + hspace), extent=bounds,
                 cmap=cm.gray, aspect=1 / 1.5)
    ax[1].set_title('Hough transform')
    ax[1].set_xlabel('Angles (degrees)')
    ax[1].set_ylabel('Distance (pixels)')
    ax[1].axis('image')

    ax[2].imshow(im_c, cmap=cm.gray)
    ax[2].set_ylim((im_c.shape[0], 0))
    ax[2].set_axis_off()
    ax[2].set_title('Detected lines')

    # You may want to change the codes below if you use a different axis choice.
    for _, dist, angle in zip(*hs_maxima):
        (x0, y0) = dist * np.array([np.cos(angle), np.sin(angle)])
        ax[2].axline((y0, x0), slope=np.tan(np.pi-angle))

    plt.tight_layout()
    plt.show()


def draw_circles(local_maxima, file_path, title):
    img = cv2.imread(file_path)
    fig, axes = plt.subplots(1, figsize=(7,7))
    axes.set_aspect('equal')
    
    axes.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    for _,r,x,y in zip(*local_maxima):
        axes.add_patch(patches.Circle((y,x),r,color=(1,0,0),fill=False)) # Circles from matplotlib.patches
    plt.title(title)
    plt.show()

    #img = cv2.imread(file_path)
    #fig = plt.figure(figsize=(7, 7))
    #plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    #circle = []
    #for _, r, x, y in zip(*local_maxima):
    #    circle.append(plt.Circle((y, x), r, color=(1, 0, 0), fill=False))
    #    fig.add_subplot(111).add_artist(circle[-1])
    #plt.title(title)
    #plt.show()
