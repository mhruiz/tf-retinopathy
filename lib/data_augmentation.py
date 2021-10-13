import imgaug.augmenters as iaa
import numpy as np


# Some helpful functions
def middle(range):
    '''
    Receives a tuple of 2 numbers (range) and returns their middle point
    '''
    return (range[0] + range[1]) / 2

def get_reduced_range(range, *percentages):
    '''
    Receives a tuple of 2 numbers (range) and 1 or 2 parameters.
    If it is 1, it will return a range reduced by the percentage given.
    If it is 2 arguments, it will return a range reduced by the percentage given for each side.
    '''
    length_range = np.abs(range[0] - range[1])
    mid = middle(range)
    mid_length = length_range / 2
    if len(percentages) == 1:
        return mid - mid_length * percentages[0], mid + mid_length * percentages[0]
    elif len(percentages) == 2:
        return mid - mid_length * percentages[0], mid + mid_length * percentages[1]
    else:
        raise Exception('Incorrect number of arguments in \'percentages\'. It must be 1 or 2 arguments')


# Data augmentation parameters
brightness_range = (-50, +50)
rotation_range = (-10, 10)
hue_offset_range = (-16, 16)

uniform_noise_range = (-10, 10)
# gaussian_noise_desv = 0 # quitar
laplacian_noise = 6
poisson_noise = 7
    

# Range for uniform noise range avoiding near 0 values
# uniform_noise_range = remove_near_zero_values(uniform_noise_range, 0.5)

# brightness_range = remove_near_zero_values(brightness_range, 0.5)

# rotation_range = remove_near_zero_values(rotation_range, 0.5)

# hue_offset_range = remove_near_zero_values(hue_offset_range, 0.5)

# ----------------
# Contrast functions
def gamma(img: np.ndarray, exponent: float):
    # Positive exponent: increase contrast on lighter areas of the image
    # Negative exponent: increase contrast on darker areas of the image
    img = img.astype(np.float32)
    return img ** (1+exponent) / 1 ** exponent

def sigmoid_limits(img: np.ndarray, alfa: float):
    img = img.astype(np.float32)
    img_ = (1/2) * (1 + (1 / np.tan((np.pi * alfa) / 2)) * np.tan(alfa * np.pi * (img/1 - 1/2)))
    return img_

def sigmoid_limits_simmetric(img: np.ndarray, alfa: float):
    # Positive alfa: increase contrast on extreme values of the image (darker and lighter)
    # Negative alfa: increase contrast on middle values of the image
    F = sigmoid_limits(img, alfa)
    
    if alfa > 0:
        return F
    else:
        I = img.astype(np.float32)
        F = F.astype(np.float32)
        F = np.clip(2*I - F, 0, 1)
 
        return F

# These are their ranges. Negative ranges will be symmetric
gamma_range = (0.35, 0.5)

sigmoid_range = (0.52, 0.68)

#--------------------

# RGB to HSI and HSI to RGB functions
# These functions were taken from this site
# https://stackoverflow.com/questions/52939362/trouble-displaying-image-hsi-converted-to-rgb-python

def rgb_to_hsi(img):
    zmax = 255 # max value
    # values in [0,1]
    R = np.divide(img[:,:,0],zmax,dtype=np.float)
    G = np.divide(img[:,:,1],zmax,dtype=np.float)
    B = np.divide(img[:,:,2],zmax,dtype=np.float)

    # Hue, when R=G=B -> H=90
    a = (0.5)*np.add(np.subtract(R,G), np.subtract(R,B)) # (1/2)*[(R-G)+(R-B)]
    b = np.sqrt(np.add(np.power(np.subtract(R,G), 2) , np.multiply(np.subtract(R,B),np.subtract(G,B))))
    tetha = np.arccos( np.divide(a, b, out=np.zeros_like(a), where=b!=0) ) # when b = 0, division returns 0, so then tetha = 90
    H = (180/np.pi)*tetha # convert rad to degree
    H[B>G]=360-H[B>G]

    # saturation = 1 - 3*[min(R,G,B)]/(R+G+B), when R=G=B -> S=0
    a = 3*np.minimum(np.minimum(R,G),B) # 3*min(R,G,B)
    b = np.add(np.add(R,G),B) # (R+G+B)
    S = np.subtract(1, np.divide(a,b,out=np.ones_like(a),where=b!=0))

    # intensity = (1/3)*[R+G+B]
    I = (1/3)*np.add(np.add(R,G),B)

    # return np.dstack((H, zmax*S, np.round(zmax*I))) # values between [0,360], [0,255] e [0,255]

    return np.dstack((H / 360.0, S, I)) # values between [0, 1], [0, 1] and [0, 1]


# A bit faster
def rgb_to_hsi_2(img):
    zmax = 255 # max value
    # values in [0,1]
    R = np.divide(img[:,:,0],zmax,dtype=np.float)
    G = np.divide(img[:,:,1],zmax,dtype=np.float)
    B = np.divide(img[:,:,2],zmax,dtype=np.float)

    # Hue, when R=G=B -> H=90
    R_minus_G = np.subtract(R,G)
    R_minus_B = np.subtract(R,B)

    a = (0.5)*np.add(R_minus_G, R_minus_B) # (1/2)*[(R-G)+(R-B)]
    b = np.sqrt(np.add(np.power(R_minus_G, 2) , np.multiply(R_minus_B,np.subtract(G,B))))
    tetha = np.arccos( np.divide(a, b, out=np.zeros_like(a), where=b!=0) ) # when b = 0, division returns 0, so then tetha = 90
    H = (180/np.pi)*tetha # convert rad to degree
 
    H[B>G]=360-H[B>G]

    # saturation = 1 - 3*[min(R,G,B)]/(R+G+B), when R=G=B -> S=0
    a = 3*np.minimum(np.minimum(R,G),B) # 3*min(R,G,B)
    b = np.add(np.add(R,G),B) # (R+G+B)
    S = np.subtract(1, np.divide(a,b,out=np.ones_like(a),where=b!=0))

    # intensity = (1/3)*[R+G+B]
    I = (1/3)*b

    # return np.dstack((H, zmax*S, np.round(zmax*I))) # values between [0,360], [0,255] e [0,255]

    return np.dstack((H, S, I)) # values between [0, 360], [0, 1] and [0, 1] -----  H is not used


def hsi_to_rgb(img):

    def f1(I,S): # I(1-S)
        return np.multiply(I, np.subtract(1,S))
    def f2(I,S,H): # I[1+(ScosH/cos(60-H))]
        r = np.pi/180
        a = np.multiply(S, np.cos(r*H)) # ScosH
        b = np.cos(r*np.subtract(60,H)) # cos(60-H)
        return np.multiply(I, np.add(1, np.divide(a,b)) )
    def f3(I,C1,C2): # 3I-(C1+C2)
        return np.subtract(3*I, np.add(C1,C2))

    zmax = 255 # max value
    # values between[0,1], [0,1] and [0,1]
    H = img[:,:,0] * 360
    # S = np.divide(img[:,:,1],zmax,dtype=np.float)
    # I = np.divide(img[:,:,2],zmax,dtype=np.float)
    S = img[:,:,1]
    I = img[:,:,2]

    R,G,B = np.ones(H.shape),np.ones(H.shape),np.ones(H.shape) # values will be between [0,1]
    # for 0 <= H < 120
    B[(0<=H)&(H<120)] = f1(I[(0<=H)&(H<120)], S[(0<=H)&(H<120)])
    R[(0<=H)&(H<120)] = f2(I[(0<=H)&(H<120)], S[(0<=H)&(H<120)], H[(0<=H)&(H<120)])
    G[(0<=H)&(H<120)] = f3(I[(0<=H)&(H<120)], R[(0<=H)&(H<120)], B[(0<=H)&(H<120)])

    # for 120 <= H < 240
    H = np.subtract(H,120)
    R[(0<=H)&(H<120)] = f1(I[(0<=H)&(H<120)], S[(0<=H)&(H<120)])
    G[(0<=H)&(H<120)] = f2(I[(0<=H)&(H<120)], S[(0<=H)&(H<120)], H[(0<=H)&(H<120)])
    B[(0<=H)&(H<120)] = f3(I[(0<=H)&(H<120)], R[(0<=H)&(H<120)], G[(0<=H)&(H<120)])

    # for 240 <= H < 360
    H = np.subtract(H,120)
    G[(0<=H)&(H<120)] = f1(I[(0<=H)&(H<120)], S[(0<=H)&(H<120)])
    B[(0<=H)&(H<120)] = f2(I[(0<=H)&(H<120)], S[(0<=H)&(H<120)], H[(0<=H)&(H<120)])
    R[(0<=H)&(H<120)] = f3(I[(0<=H)&(H<120)], G[(0<=H)&(H<120)], B[(0<=H)&(H<120)])

    # # Round
    # RGB = np.round(np.dstack( ((zmax*R) , (zmax*G) , (zmax*B)) ))
    
    # # Clip to [0, 255] and convert to uint8
    # RGB = np.clip(RGB, 0, 255).astype(np.uint8)
    
    RGB = np.dstack( (R, G, B) )
    RGB = np.clip(RGB, 0, 1)

    return RGB

# A bit faster
def hsi_to_rgb_2(img):

    def f1(I,S): # I(1-S)
        return np.multiply(I, np.subtract(1,S))
    def f2(I,S,H): # I[1+(ScosH/cos(60-H))]
        r = np.pi/180
        a = np.multiply(S, np.cos(r*H)) # ScosH
        b = np.cos(r*np.subtract(60,H)) # cos(60-H)
        return np.multiply(I, np.add(1, np.divide(a,b)) )
    def f3(I,C1,C2): # 3I-(C1+C2)
        return np.subtract(3*I, np.add(C1,C2))

    zmax = 255 # max value
    # values between [0,360], [0,1] and [0,1]
    H = img[:,:,0] 
    # S = np.divide(img[:,:,1],zmax,dtype=np.float)
    # I = np.divide(img[:,:,2],zmax,dtype=np.float)
    S = img[:,:,1]
    I = img[:,:,2]

    R,G,B = np.ones(H.shape),np.ones(H.shape),np.ones(H.shape) # values will be between [0,1]

    H_binary = (0<=H)&(H<120)

    # for 0 <= H < 120
    B[H_binary] = f1(I[H_binary], S[H_binary])
    R[H_binary] = f2(I[H_binary], S[H_binary], H[H_binary])
    G[H_binary] = f3(I[H_binary], R[H_binary], B[H_binary])

    # for 120 <= H < 240
    H = np.subtract(H,120)
    H_binary = (0<=H)&(H<120)
    R[H_binary] = f1(I[H_binary], S[H_binary])
    G[H_binary] = f2(I[H_binary], S[H_binary], H[H_binary])
    B[H_binary] = f3(I[H_binary], R[H_binary], G[H_binary])

    # for 240 <= H < 360
    H = np.subtract(H,120)
    H_binary = (0<=H)&(H<120)
    G[H_binary] = f1(I[H_binary], S[H_binary])
    B[H_binary] = f2(I[H_binary], S[H_binary], H[H_binary])
    R[H_binary] = f3(I[H_binary], G[H_binary], B[H_binary])

    # # Round
    # RGB = np.round(np.dstack( ((zmax*R) , (zmax*G) , (zmax*B)) ))
    
    # # Clip to [0, 255] and convert to uint8
    # RGB = np.clip(RGB, 0, 255).astype(np.uint8)
    
    RGB = np.dstack( (R, G, B) )
    RGB = np.clip(RGB, 0, 1)

    return RGB


# -----------------------------------

# This function permits call different contrast functions with different ranges
def apply_contrast_transformation(images, random_state, function, range_values, sign):
    '''
    Receives a batch of images as a list
    A np.random.RandomState object to perform random operations
    A function to perform contrast changes
    A range of values. It must be specified as (positive min, positive max)
    An integer 'sign' whose value will make the function uses full symmetric range or not
    - 0: only positive range
    - 1: only negative range
    - 2: full symmetric range
    Returns a batch of transformed images
    '''

    # Get random value a function parameter
    # Choose range
    if sign == 0:
        # Positive values --> Keeps range
        choosen_range = range_values
    elif sign == 1:
        # Negative values --> Inverted range and multiplied by -1
        choosen_range = (-range_values[1], -range_values[0])
    else:
        # Positive range if random value is lower than 0.5, else negative range
        choosen_range = range_values if random_state.rand(1)[0] < 0.5 else (-range_values[1], -range_values[0])
    
    def get_random_value_in_range(rng):
        # Rng: (min, max)
        
        # (Max - Min) * U(0, 1) + Min --> U(Min, Max)
        return (rng[1] - rng[0]) * random_state.rand(1)[0] + rng[0]

    random_argument_value = get_random_value_in_range(choosen_range)

    # Choose color channels where function will work
    # Randomly choose between HSI (I, S, SI) or RGB (RGB, RG) # Removed RG
    # HSI channels will have '-1' as identifier
    # RGB, '-2'
    hsi_channels = [[-1] + x for x in [[2], [1], [1,2]] ]
    rgb_channels = [[-2] + x for x in [[0,1,2] ]]
    color_channels = hsi_channels + rgb_channels

    # Get channels
    color_channel = color_channels[random_state.randint(0, len(color_channels))]
    # color_channels = [[-1, 2], [-1, 1], [-1, 1, 2], [-2, 0, 1, 2]]

    # Transformed images
    transformed_batch = []

    for i in range(len(images)):
        current_image = images[i]

        if color_channel[0] == -1:
            # Get HSI normalized image
            prepared_image = rgb_to_hsi(current_image)
        else:
            # Normalize RGB 
            prepared_image = current_image.astype(np.float) / 255.0

        # Apply contrast function on each channel
        transformed_img = np.copy(prepared_image)
        for ch in color_channel[1:]:
            transformed_img[:,:,ch] = function(prepared_image[:,:,ch], random_argument_value)

        # Convert it back to [0,255] RGB image
        if color_channel[0] == -1:
            transformed_img = np.round(hsi_to_rgb(transformed_img) * 255).astype(np.uint8)
        else:
            transformed_img = np.round(transformed_img * 255).astype(np.uint8)

        # Copy image to new batch
        transformed_batch.append(transformed_img)

    return transformed_batch

# A bit faster: 1s lower per 200 images
def apply_contrast_transformation_2(images, random_state, function, range_values):
    '''
    Receives a batch of images as a list
    A np.random.RandomState object to perform random operations
    A function to perform contrast changes
    A range of values. It must be specified as a tuple -> (positive min, positive max)

    Returns a batch of transformed images
    '''

    # Always full range
    # Positive range if random value is lower than 0.5, else negative range
    choosen_range = range_values if random_state.rand(1)[0] < 0.5 else (-range_values[1], -range_values[0])

    random_argument_value = (choosen_range[1] - choosen_range[0]) * random_state.rand(1)[0] + choosen_range[0]

    # Choose color channels where function will work
    # Randomly choose between HSI (I, S, SI) or RGB (RGB, RG) # Removed RG
    # HSI channels will have '-1' as identifier
    # RGB, '-2'
    hsi_channels = [[-1] + x for x in [[2], [1], [1,2]] ]
    rgb_channels = [[-2] + x for x in [[0,1,2] ]]
    color_channels = hsi_channels + rgb_channels

    # Get channels
    color_channel = color_channels[random_state.randint(0, len(color_channels))]
    # color_channels = [[-1, 2], [-1, 1], [-1, 1, 2], [-2, 0, 1, 2]]

    # Transformed images
    transformed_batch = []

    for i in range(len(images)):
        current_image = images[i]

        if color_channel[0] == -1:
            # Get HSI normalized image
            prepared_image = rgb_to_hsi_2(current_image)
        else:
            # Normalize RGB 
            prepared_image = current_image.astype(np.float) / 255.0

        # Apply contrast function on each channel
        for ch in color_channel[1:]:
            prepared_image[:,:,ch] = function(prepared_image[:,:,ch], random_argument_value)

        # Convert it back to [0,255] RGB image
        if color_channel[0] == -1:
            prepared_image = np.round(hsi_to_rgb_2(prepared_image) * 255).astype(np.uint8)
        else:
            prepared_image = np.round(prepared_image * 255).astype(np.uint8)

        # Copy image to new batch
        transformed_batch.append(prepared_image)

    return transformed_batch

#---------------

# These are the augmenter objects which will be used to perform data-augmentation

# Contrast transformation: gamma and sigmoid functions have both 50% of probability. They will be applied with full range
contrast_augmenter = iaa.OneOf([
            iaa.Lambda(func_images=lambda images, random_state, parents, hooks: 
                                        apply_contrast_transformation_2(images, random_state, gamma, gamma_range)),
            iaa.Lambda(func_images=lambda images, random_state, parents, hooks: 
                                        apply_contrast_transformation_2(images, random_state, sigmoid_limits_simmetric, sigmoid_range))
])

# Noise
# Se ha quitado el ruido gaussiano porque cuando la desviacion tipica es menor que 8, parece no aplicar cambios
# y cuando es mayor o igual a 8, el cambio es excesivo
# En su lugar, habrá un 25% de probabilidad (lo que correspondería al Gaussiano) de que no se aplique ninguna operación de ruido
# Por eso el 'iaa.Sometimes(0.75, ...)'

noise_aug = iaa.Sometimes(0.75, iaa.OneOf([
    iaa.OneOf([
        # Valores negativos
        iaa.AddElementwise((uniform_noise_range[0], get_reduced_range(uniform_noise_range, 0.5)[0]), per_channel=True),
        # Valores positivos
        iaa.AddElementwise((get_reduced_range(uniform_noise_range, 0.5)[1], uniform_noise_range[1]), per_channel=True),
    ]),
    iaa.AdditiveLaplaceNoise(scale=(0, laplacian_noise), per_channel=True),
    iaa.AdditivePoissonNoise((0, poisson_noise), per_channel=True)
]))

# Previous augmenter
aug = iaa.Sequential([
    iaa.OneOf([
        noise_aug,
        # Al 75% de las imagenes se les aplica alguna de las simetrías
        iaa.Fliplr(1),
        iaa.Flipud(1),
        # Rotation
        iaa.OneOf([
            iaa.Affine(rotate=(rotation_range[0], int(get_reduced_range(rotation_range, 0.5)[0]))),
            iaa.Affine(rotate=(int(get_reduced_range(rotation_range, 0.5)[1]), rotation_range[1]))
        ]),
        # HUE
        iaa.OneOf([
            iaa.AddToHue(value=(hue_offset_range[0], int(get_reduced_range(hue_offset_range, 0.5)[0]))),
            iaa.AddToHue(value=(int(get_reduced_range(hue_offset_range, 0.5)[1]), hue_offset_range[1]))
        ])
    ]),
    # After all, apply jigsaw transformation
    iaa.Sometimes(0.8, iaa.Jigsaw(nb_rows=2, nb_cols=2)) # 8 de cada 10 aprox
])


# Alternative augmenter
augmenter = iaa.Sequential([
    iaa.OneOf([
        # Add noise
        noise_aug, # There is a 25% of ptobability that the image keeps without any changes. 1/6 * 25% 
        iaa.Fliplr(1),
        iaa.Flipud(1),
        # Rotation
        iaa.OneOf([
            # Negative values
            iaa.Affine(rotate=(rotation_range[0], int(get_reduced_range(rotation_range, 0.5)[0]))),
            # Positive values
            iaa.Affine(rotate=(int(get_reduced_range(rotation_range, 0.5)[1]), rotation_range[1]))
        ]),
        # HUE
        iaa.OneOf([
            # Negative values
            iaa.AddToHue(value=(hue_offset_range[0], int(get_reduced_range(hue_offset_range, 0.5)[0]))),
            # Positive values
            iaa.AddToHue(value=(int(get_reduced_range(hue_offset_range, 0.5)[1]), hue_offset_range[1]))
        ]),
        # Contrast
        contrast_augmenter
        # No brightness transformation because gamma can change it
    ]),
    # After all, apply jigsaw transformation
    iaa.Sometimes(0.8, iaa.Jigsaw(nb_rows=2, nb_cols=2)) # 8 de cada 10 aprox
])


def py_apply_data_augmentation(image: np.ndarray):
    # return augmenter.augment_image(image)
    # New data augmenter
    return aug.augment_image(image)

def py_apply_data_augmentation_2(image: np.ndarray):
    # New new augmenter
    return augmenter.augment_image(image)

