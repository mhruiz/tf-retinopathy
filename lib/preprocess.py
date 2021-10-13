from pylab import array, arange, uint8
import numpy as np
import cv2

########################################################################
# Functions from Voets et all 2019 GitHub

def _increase_contrast(image):
    """
    Helper function for increasing contrast of image.
    """
    # Create a local copy of the image.
    copy = image.copy()

    maxIntensity = 255.0
    x = arange(maxIntensity)

    # Parameters for manipulating image data.
    phi = 1.3
    theta = 1.5
    y = (maxIntensity/phi)*(x/(maxIntensity/theta))**0.5

    # Decrease intensity such that dark pixels become much darker,
    # and bright pixels become slightly dark.
    copy = (maxIntensity/phi)*(copy/(maxIntensity/theta))**2
    copy = array(copy, dtype=uint8)

    return copy

def _find_contours(image):
    """
    Helper function for finding contours of image.

    Returns coordinates of contours.
    """
    # Increase constrast in image to increase changes of finding
    # contours.
    processed = _increase_contrast(image)

    # Get the gray-scale of the image.
    gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)

    # Detect contour(s) in the image.
    cnts = cv2.findContours(
        gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    center = None

    # At least ensure that some contours were found.
    if len(cnts) > 0:
        # Find the largest contour in the mask.
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)

        # Assume the radius is of a certain size.
        if radius > 100:
            M = cv2.moments(c)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            return (center, radius)

##################################################################################
# Own functions

def get_data_image(img_path):
    '''
    Read the image given by its path (img_path) and try to detect if there is any contout inside.
    Return five objects, which are: size of image in x and y dimmensions, radius of the detected contour and the x and y coordinates of the contour center
    (size_x, size_y, radius, pos_center_x, pos_center_y).
    If the image has no contour, radius, pos_center_x and pos_center_x will return -1
    '''
    # Read image
    img = cv2.imread(img_path, -1)

    # Image size
    s_y = img.shape[0]
    s_x = img.shape[1]

    # Check if contour is detected
    contour = _find_contours(img)
    if contour is None:
        # If no contour has been detected, write -1
        pos_center_x = pos_center_y = radius = -1
    else:
        center, radius = contour
        pos_center_x, pos_center_y = center
    
    return s_x, s_y, radius, pos_center_x, pos_center_y

def get_info_all_images_chunk(data):
    '''
    This function receives a DataFrame object (data) which must have a 'path' column.
    Reads row by row the 'path' column and process each image path.
    Returns five lists, in a tuple, which are: size of image in x and y dimmensions, radius of the detected contour and the x and y coordinates of the contour center
    (size_x, size_y, radius, pos_center_x, pos_center_y)
    '''
    # Create lists
    size_x = []
    size_y = []
    radius = []
    pos_center_x = []
    pos_center_y = []
    # For each row in the DataFrame given
    for row in data.itertuples():
        # Get data of image
        v1, v2, v3, v4, v5 = get_data_image(row.path)
        # Save to lists
        size_x.append(v1)
        size_y.append(v2)
        radius.append(v3)
        pos_center_x.append(v4)
        pos_center_y.append(v5)
    return (size_x, size_y, radius, pos_center_x, pos_center_y)

def crop_image(img, radius, c_x, c_y):
    '''
    Crops the area of the image specified by the radius and center (c_x, c_y) given.
    Then, fills the cropped image to obtain a square shape image, keeping the center
    of the retina at the center of the image
    '''
    radius = np.round(radius)

    # Locate retina
    t_lim = c_y - radius
    b_lim = c_y + radius
    l_lim = c_x - radius
    r_lim = c_x + radius

    # Index positions in x and y
    y = np.arange(t_lim, b_lim + 1)
    x = np.arange(l_lim, r_lim + 1)

    # Crop retina
    t_lim = int(np.maximum(t_lim, 0))
    b_lim = int(np.minimum(b_lim + 1, img.shape[0]))

    l_lim = int(np.maximum(l_lim, 0))
    r_lim = int(np.minimum(r_lim + 1, img.shape[1]))

    crop = img[t_lim:b_lim,l_lim:r_lim,:]

    # Define borders - Get number of pixels missing in crop in each side
    top_b = np.count_nonzero(y < 0)
    bottom_b = np.count_nonzero(y >= img.shape[0])
    left_b = np.count_nonzero(x < 0)
    right_b = np.count_nonzero(x >= img.shape[1])

    # Create new black square shape image
    centered = np.zeros((top_b + crop.shape[0] + bottom_b, left_b + crop.shape[1] + right_b, 3),dtype=np.uint8)
    # Place retina image centered
    centered[top_b:centered.shape[0]-bottom_b, left_b:centered.shape[1]-right_b, :] = crop

    return centered

def crop_and_resize_image(img, radius, c_x, c_y, size):
    '''
    Crops the area of the image specified by the radius and center (c_x, c_y) given.
    Resizes the cropped image to have a retina diametrer equals to 'size'.
    Then, fills the cropped image to obtain a squared shape image, keeping the center
    of the retina at the center of the image
    '''
    radius_f = radius
    radius = np.round(radius)

    # Locate retina - Get its 4 theoretical limits
    t_lim = c_y - radius
    b_lim = c_y + radius
    l_lim = c_x - radius
    r_lim = c_x + radius

    # Index positions in x and y
    y = np.arange(t_lim, b_lim + 1)
    x = np.arange(l_lim, r_lim + 1)

    # Crop retina - Get its 4 limits
    t_lim = int(np.maximum(t_lim, 0))
    b_lim = int(np.minimum(b_lim + 1, img.shape[0]))

    l_lim = int(np.maximum(l_lim, 0))
    r_lim = int(np.minimum(r_lim + 1, img.shape[1]))

    # Get the cropped image - only the retina
    crop = img[t_lim:b_lim,l_lim:r_lim,:]

    # Define borders - Get number of pixels missing in crop for each side (in original image)
    top_b = np.count_nonzero(y < 0)
    bottom_b = np.count_nonzero(y >= img.shape[0])
    left_b = np.count_nonzero(x < 0)
    right_b = np.count_nonzero(x >= img.shape[1])

    # Scale factor to resize cropped image making the retina have, as diameter, the value of 'size'
    f = size / (radius * 2)

    # Check that resized retina will not be greater than size
    big_shape = int(np.max(crop.shape))
    big_resized_shape = int(np.ceil(big_shape * f))
    if big_resized_shape > size:
        # Adjust f for avoiding retina being bigger than spicified size
        f = size / big_shape

    # Resize cropped image
    crop = cv2.resize(crop, (0,0), fx=f, fy=f)

    # Get missing pixels for each dimmension
    missing_vertical = size - crop.shape[0]
    missing_horizontal = size - crop.shape[1]

    proportion_top = top_b / (top_b + bottom_b) if missing_vertical != 0 else 0
    proportion_left = left_b / (left_b + right_b) if missing_horizontal != 0 else 0

    # Set border size for new size (scaled image)
    top_b = int(np.floor(proportion_top * missing_vertical))
    bottom_b = int(np.ceil((1-proportion_top) * missing_vertical))

    left_b = int(np.floor(proportion_left * missing_horizontal))
    right_b = int(np.ceil((1-proportion_left) * missing_horizontal))

    # If any pixel is missing in image's height, add it to the bottom border
    if top_b + crop.shape[0] + bottom_b < size:
        bottom_b += (size - (top_b + crop.shape[0] + bottom_b))

    # If any pixel is missing in image's width, add it to the right border
    if left_b + crop.shape[1] + right_b < size:
        right_b += (size - (left_b + crop.shape[1] + right_b))

    # Create new black squared shape image
    centered = np.zeros((top_b + crop.shape[0] + bottom_b, left_b + crop.shape[1] + right_b, 3),dtype=np.uint8)

    # Place retina image centered
    centered[top_b:centered.shape[0]-bottom_b, left_b:centered.shape[1]-right_b, :] = crop

    return centered

def process_image(img_name, img_path, save_path, radius, c_x, c_y, size):
    ''' 
    Reads the image given by 'img_path'.
    Crops using the radius and center position and pads to get a squared shape, making its center coincide with the center of the retina,
    then, resizes it to the given 'size'.
    Saves the new image to 'save_path' with same 'img_name'' and .png as file extension.
    Returns the path to the new processed image
    '''
    img = cv2.imread(img_path, -1)

    img_new = crop_and_resize_image(img, radius, c_x, c_y, size)

    new_path = save_path + img_name + '.png' # All processed images will be .png
    
    cv2.imwrite(new_path, img_new)

    return new_path
