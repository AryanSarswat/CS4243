import os
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
import math

##### Part 1: image preprossessing #####

def rgb2gray(img):
    """
    5 points
    Convert a colour image greyscale
    Use (R,G,B)=(0.299, 0.587, 0.114) as the weights for red, green and blue channels respectively
    :param img: numpy.ndarray (dtype: np.uint8)
    :return img_gray: numpy.ndarray (dtype:np.uint8)
    """
    if len(img.shape) != 3:
        print('RGB Image should have 3 channels')
        return
    
    ###Your code here###
    height, width = img.shape[:2]
    RGB2GRAY = lambda RGB : 0.299 * RGB[0] + 0.587 * RGB[1] + 0.114 * RGB[2]
    img_gray = np.zeros(img.shape[:2])
    for h in range(height):
        for w in range(width):
            img_gray[h][w] = np.round(RGB2GRAY(img[h][w]))
    ###
    img_gray = img_gray.astype('uint8')
    return img_gray

def convolve(img, kernel, k):
    """
    Performs a convolution operation on a image img and kernel of size k. It pads the original image with 0's so that the return image is the same size as the original image.
    :param img: numpy.ndarray
    :param kernel: numpy.ndarray
    :param k size of kernel: int
    :return convolved_image: numpy.ndarray
    """
    padded_image = pad_zeros(img, k, k, k, k)
    height, width = img.shape[:2]
    convolved_image = img.astype('float')
    for h in range(k, height + k):
        for w in range(k, width + k):
            new_pixel = 0.0
            for a in range(-k,k+1,1):
                for b in range(-k, k+1, 1):
                    new_pixel += padded_image[h + a][w + b] * kernel[k - a][k - b]
            convolved_image[h - k][w - k] = new_pixel
    return convolved_image

def gray2grad(img):
    """
    5 points
    Estimate the gradient map from the grayscale images by convolving with Sobel filters (horizontal and vertical gradients) and Sobel-like filters (gradients oriented at 45 and 135 degrees)
    The coefficients of Sobel filters are provided in the code below.
    :param img: numpy.ndarray
    :return img_grad_h: horizontal gradient map. numpy.ndarray
    :return img_grad_v: vertical gradient map. numpy.ndarray
    :return img_grad_d1: diagonal gradient map 1. numpy.ndarray
    :return img_grad_d2: diagonal gradient map 2. numpy.ndarray
    """
    sobelh = np.array([[-1, 0, 1], 
                       [-2, 0, 2], 
                       [-1, 0, 1]], dtype = float)
    sobelv = np.array([[-1, -2, -1], 
                       [0, 0, 0], 
                       [1, 2, 1]], dtype = float)
    sobeld1 = np.array([[-2, -1, 0],
                        [-1, 0, 1],
                        [0,  1, 2]], dtype = float)
    sobeld2 = np.array([[0, -1, -2],
                        [1, 0, -1],
                        [2, 1, 0]], dtype = float)
    
    ###Your code here####
    img_grad = img.astype('float')
    img_grad_h = convolve(img_grad, sobelh , 1)
    img_grad_v = convolve(img_grad, sobelv, 1)
    img_grad_d1 = convolve(img_grad, sobeld1, 1)
    img_grad_d2 = convolve(img_grad, sobeld2, 1)
    ###
    return img_grad_h, img_grad_v, img_grad_d1, img_grad_d2

def pad_zeros(img, pad_height_bef, pad_height_aft, pad_width_bef, pad_width_aft):
    """
    5 points
    Add a border of zeros around the input images so that the output size will match the input size after a convolution or cross-correlation operation.
    e.g., given matrix [[1]] with pad_height_bef=1, pad_height_aft=2, pad_width_bef=3 and pad_width_aft=4, obtains:
    [[0 0 0 0 0 0 0 0]
    [0 0 0 1 0 0 0 0]
    [0 0 0 0 0 0 0 0]
    [0 0 0 0 0 0 0 0]]
    :param img: numpy.ndarray
    :param pad_height_bef: int
    :param pad_height_aft: int
    :param pad_width_bef: int
    :param pad_width_aft: int
    :return img_pad: numpy.ndarray. dtype is the same as the input img. 
    """
    height, width = img.shape[:2]
    new_height, new_width = (height + pad_height_bef + pad_height_aft), (width + pad_width_bef + pad_width_aft)
    img_pad = np.zeros((new_height, new_width)) if len(img.shape) == 2 else np.zeros((new_height, new_width, img.shape[2]))

    ###Your code here###
    for h in range(height):
        for w in range(width):
             img_pad[h + pad_height_bef][w + pad_width_bef] = img[h][w]
    return img_pad.astype(img.dtype)

##### Part 2: Normalized Cross Correlation #####
def normalized_cross_correlation(img, template):
    """
    10 points.
    Implement the cross-correlation operation in a naive 6 nested for-loops. 
    The 6 loops include the height, width, channel of the output and height, width and channel of the template.
    :param img: numpy.ndarray.
    :param template: numpy.ndarray.
    :return response: numpy.ndarray. dtype: float
    """
    Hi, Wi = img.shape[:2]
    Hk, Wk = template.shape[:2]
    Ho = Hi - Hk + 1
    Wo = Wi - Wk + 1

    ###Your code here###
    
    # Place target element on top left corner of kernel instead of center

    # Use floats for img and template, otherwise uint8 overflow in later operations
    img_f = img.astype('float')
    template_f = template.astype('float')

    # Alternative: np.linalg.norm(template_f)
    kernel_mag = math.sqrt(np.sum(template_f**2))
    response = np.zeros((Ho, Wo))

    # GREYSCALE
    if len(img.shape) == 2:
        for i in range(Ho):
            for j in range(Wo):
                covered_elements = img_f[i:i+Hk, j:j+Wk]
                covered_mag = math.sqrt(np.sum(covered_elements**2))
                norm_coefficient = 1 / (kernel_mag * covered_mag)
                filtered_val = 0.0
            
                for k in range(Hk):
                    for l in range(Wk):
                        neighbour = img_f[i+k, j+l]
                        filtered_val += (template_f[k, l] * neighbour)

                response[i, j] = norm_coefficient * filtered_val

    #RGB
    elif len(img.shape) == 3:
        for i in range(Ho):
            for j in range(Wo):
                covered_elements = img_f[i:i+Hk, j:j+Wk, :]
                covered_mag = math.sqrt(np.sum(covered_elements**2))
                norm_coefficient = 1 / (kernel_mag * covered_mag)
                filtered_val = 0.0
            
                for c in range(3):
                    for k in range(Hk):
                        for l in range(Wk):
                            for ct in range(3):
                                # Can remove one of the channel loops & the continue block, left in for 6 loops
                                if c != ct:
                                    continue
                                neighbour = img_f[i+k, j+l, c]
                                filtered_val += (template_f[k, l, ct] * neighbour)

                response[i, j] = norm_coefficient * filtered_val
                
    ###
    return response


def normalized_cross_correlation_fast(img, template):
    """
    10 points.
    Implement the cross correlation with 3 nested for-loops. 
    The for-loop over the template is replaced with the element-wise multiplication between the kernel and the image regions.
    :param img: numpy.ndarray
    :param template: numpy.ndarray
    :return response: numpy.ndarray. dtype: float
    """
    Hi, Wi = img.shape[:2]
    Hk, Wk = template.shape[:2]
    Ho = Hi - Hk + 1
    Wo = Wi - Wk + 1

    ###Your code here###
    img_f = img.astype('float')
    template_f = template.astype('float')

    # Alternative: np.linalg.norm(template_f)
    kernel_mag = math.sqrt(np.sum(template_f**2))
    response = np.zeros((Ho, Wo))

    # GREYSCALE
    if len(img.shape) == 2:
        for i in range(Ho):
            for j in range(Wo):
                covered_elements = img_f[i:i+Hk, j:j+Wk]
                covered_mag = math.sqrt(np.sum(covered_elements**2))
                norm_coefficient = 1 / (kernel_mag * covered_mag)
                filtered_val = np.sum(template_f * covered_elements)
                response[i, j] = norm_coefficient * filtered_val

    #RGB
    elif len(img.shape) == 3:
        for i in range(Ho):
            for j in range(Wo):
                covered_elements = img_f[i:i+Hk, j:j+Wk, :]
                covered_mag = math.sqrt(np.sum(covered_elements**2))
                norm_coefficient = 1 / (kernel_mag * covered_mag)
                filtered_val = 0.0
            
                for c in range(3):
                    neighbours = covered_elements[:, :, c]
                    filtered_val += np.sum(template_f[:, :, c] * neighbours)

                # Can remove channel loop and directly use covered_elements & templates 3D array, left in for 3 loops
                # filtered_val += np.sum(template_f * covered_elements)
            
                response[i, j] = norm_coefficient * filtered_val

    ###
    return response




def normalized_cross_correlation_matrix(img, template):
    """
    10 points.
    Converts cross-correlation into a matrix multiplication operation to leverage optimized matrix operations.
    Please check the detailed instructions in the pdf file.
    :param img: numpy.ndarray
    :param template: numpy.ndarray
    :return response: numpy.ndarray. dtype: float
    """
    Hi, Wi = img.shape[:2]
    Hk, Wk = template.shape[:2]
    Ho = Hi - Hk + 1
    Wo = Wi - Wk + 1

    ###Your code here###
    img_f = img.astype('float')
    template_f = template.astype('float')

    # Alternative: np.linalg.norm(template_f)
    kernel_mag = math.sqrt(np.sum(template_f**2))
    num_channels = 3 if len(img.shape) == 3 else 1

    template_r = np.zeros((num_channels*Hk*Wk, 1))    
    input_r = np.zeros((Ho*Wo, num_channels*Hk*Wk))
    reshape_col_count = 0

    ### Matrix reshaping

    # GREYSCALE
    if len(img.shape) == 2:
        template_r = [[t] for t in template.flatten()]
        # template_r = np.reshape(template, (Hk*Wk, 1))
        for i in range(Hk):
            for j in range(Wk):
                input_r[:, reshape_col_count] = img_f[i:i+Ho, j:j+Wo].flatten()
                reshape_col_count += 1

    # RGB
    elif len(img.shape) == 3:
        template_r_len = Hk*Wk
        template_r[:template_r_len, 0] = template[:, :, 0].flatten()
        template_r[template_r_len:2*template_r_len, 0] = template[:, :, 1].flatten()
        template_r[2*template_r_len:, 0] = template[:, :, 2].flatten()
        
        for c in range(3):
            for i in range(Hk):
                for j in range(Wk):
                    input_r[:, reshape_col_count] = img_f[i:i+Ho, j:j+Wo, c].flatten()
                    reshape_col_count += 1
        
    #####################

    # Magnitude of covers matrix
    ones_kernel = np.ones((num_channels*Hk*Wk, 1))
    input_squared = input_r**2
    cover_square_sums = np.matmul(input_squared, ones_kernel)
    cover_mags = np.sqrt(cover_square_sums)

    # Calculate final results
    response_r = np.matmul(input_r, template_r) # or numpy.dot
    response_initial = (1 / (kernel_mag * cover_mags)) * response_r
    # Reshape the Ho*Wo by 1 matrix into Ho by Wo matrix
    response = np.reshape(response_initial, (Ho, Wo))

    ###
    return response


##### Part 3: Non-maximum Suppression #####

def non_max_suppression(response, suppress_range, threshold=None):
    """
    10 points
    Implement the non-maximum suppression for translation symmetry detection
    The general approach for non-maximum suppression is as follows:
	1. Set a threshold τ; values in X<τ will not be considered.  Set X<τ to 0.  
    2. While there are non-zero values in X
        a. Find the global maximum in X and record the coordinates as a local maximum.
        b. Set a small window of size w×w points centered on the found maximum to 0.
	3. Return all recorded coordinates as the local maximum.
    :param response: numpy.ndarray, output from the normalized cross correlation
    :param suppress_range: a tuple of two ints (H_range, W_range). 
                           the points around the local maximum point within this range are set as 0. In this case, there are 2*H_range*2*W_range points including the local maxima are set to 0
    :param threshold: int, points with value less than the threshold are set to 0
    :return res: a sparse response map which has the same shape as response
    """
    ###Your code here###
    res = np.where(response < threshold, 0 , response) #Set all values below a threshold to 0
    height, width = res.shape[:2]
    H_range, W_range = suppress_range
    max_coord = []
    while (np.count_nonzero(res) > 0):
        global_max = np.max(res)
        max_loc = np.where(res == global_max)
        max_loc = (max_loc[0][0], max_loc[1][0])
        max_coord.append(max_loc)

        H_low = max_loc[0] - int(H_range/2)
        H_high = max_loc[0] + int(H_range/2) 
        if H_low < 0 :
            H_low = 0
        if H_high >= height:
            H_high = height

        W_low = max_loc[1] - int(W_range/2) 
        W_high = max_loc[1] + int(W_range/2)
        if W_low < 0 :
                W_low = 0
        if W_high >= width:
            W_high = width
        res[H_low:H_high, W_low:W_high] = np.zeros((H_high - H_low, W_high - W_low))
    ###
    for i in max_coord:
        res[i[0],i[1]] = 1
    return res

##### Part 4: Question And Answer #####
def normalized_cross_correlation_ms(img, template):
    """
    10 points
    Please implement mean-subtracted cross correlation which corresponds to OpenCV TM_CCOEFF_NORMED.
    For simplicty, use the "fast" version.
    :param img: numpy.ndarray
    :param template: numpy.ndarray
    :return response: numpy.ndarray. dtype: float
    """
    Hi, Wi = img.shape[:2]
    Hk, Wk = template.shape[:2]
    Ho = Hi - Hk + 1
    Wo = Wi - Wk + 1

    ###Your code here###
    img_f = img.astype('float')
    template_f = template.astype('float')

    # Alternative: np.linalg.norm(template_f)
    kernel_ms = template_f - np.mean(template_f)
    kernel_mag = np.linalg.norm(kernel_ms)
    response = np.zeros((Ho, Wo))

    # GREYSCALE
    if len(img.shape) == 2:
        for i in range(Ho):
            for j in range(Wo):
                covered_elements = img_f[i:i+Hk, j:j+Wk]
                covered_elements_ms = covered_elements - np.mean(covered_elements)
                covered_mag = np.linalg.norm(covered_elements_ms)
                norm_coefficient = 1 / (kernel_mag * covered_mag)
                filtered_val = np.sum(kernel_ms * covered_elements_ms)
                response[i, j] = norm_coefficient * filtered_val

    #RGB
    elif len(img.shape) == 3:
        for i in range(Ho):
            for j in range(Wo):
                covered_elements = img_f[i:i+Hk, j:j+Wk, :]
                R_mean = np.mean(covered_elements[:, :, 0])
                G_mean = np.mean(covered_elements[:, :, 1])
                B_mean = np.mean(covered_elements[:, :, 2])
                covered_elements_ms = covered_elements - [R_mean, G_mean, B_mean]
                covered_mag = math.sqrt(np.sum(covered_elements_ms**2))
                norm_coefficient = 1 / (kernel_mag * covered_mag)
                filtered_val = 0.0
            
                for c in range(3):
                    neighbours = covered_elements_ms[:, :, c]
                    filtered_val += np.sum(kernel_ms[:, :, c] * neighbours)
                # Can remove channel loop and directly use covered_elements & templates 3D array, left in for 3 loops
                # filtered_val += np.sum(template_f * covered_elements)
                response[i, j] = norm_coefficient * filtered_val
    ###
    return response






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

def show_imgs(imgs, titles=None):
    '''
    Display a list of images in the notebook cell.
    :param imgs: a list of images or a single image
    '''
    if isinstance(imgs, list) and len(imgs) != 1:
        n = len(imgs)
        fig, axs = plt.subplots(1, n, figsize=(15,15))
        for i in range(n):
            axs[i].imshow(imgs[i], cmap='gray' if len(imgs[i].shape) == 2 else None)
            if titles is not None:
                axs[i].set_title(titles[i])
    else:
        img = imgs[0] if (isinstance(imgs, list) and len(imgs) == 1) else imgs
        plt.figure()
        plt.imshow(img, cmap='gray' if len(img.shape) == 2 else None)

def show_img_with_squares(response, img_ori=None, rec_shape=None):
    '''
    Draw small red rectangles of size defined by rec_shape around the non-zero points in the image.
    Display the rectangles and the image with rectangles in the notebook cell.
    :param response: numpy.ndarray. The input response should be a very sparse image with most of points as 0.
                     The response map is from the non-maximum suppression.
    :param img_ori: numpy.ndarray. The original image where response is computed from
    :param rec_shape: a tuple of 2 ints. The size of the red rectangles.
    '''
    response = response.copy()
    if img_ori is not None:
        img_ori = img_ori.copy()
    H, W = response.shape[:2]
    if rec_shape is None:
        h_rec, w_rec = 25, 25
    else:
        h_rec, w_rec = rec_shape

    xs, ys = response.nonzero()
    for x, y in zip(xs, ys):
        response = cv2.rectangle(response, (y - h_rec//2, x - w_rec//2), (y + h_rec//2, x + w_rec//2), (255, 0, 0), 2)
        if img_ori is not None:
            img_ori = cv2.rectangle(img_ori, (y - h_rec//2, x - w_rec//2), (y + h_rec//2, x + w_rec//2), (0, 255, 0), 2)
        
    if img_ori is not None:
        show_imgs([response, img_ori])
    else:
        show_imgs(response)