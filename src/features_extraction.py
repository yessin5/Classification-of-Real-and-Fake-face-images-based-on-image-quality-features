import torch
import cv2
import numpy as np
import pyiqa
import os

def dtype_limits(image, clip_negative=False):
    """Return intensity limits,(min, max) tuple, of the image's dtype"""
    _integer_types = (np.byte, np.ubyte, np.short, np.ushort, np.intc, np.uintc, int, np.int_, np.uint, np.longlong, np.ulonglong)
    _integer_ranges = {t: (np.iinfo(t).min, np.iinfo(t).max) for t in _integer_types}
    dtype_range = {bool: (False, True), np.bool_: (False, True), np.bool8: (False, True), float: (-1, 1), np.float_: (-1, 1), np.float16: (-1, 1), np.float32: (-1, 1), np.float64: (-1, 1)}
    dtype_range.update(_integer_ranges)

    imin, imax = dtype_range[image.dtype.type]
    if clip_negative:
        imin = 0
    return imin, imax

def contrast_score(image_param, lower_percentile=1, upper_percentile=99, method='linear'):
    """Calculate the contrast score of an image"""
    # Convert the image to a NumPy array if it's a CUDA tensor
    if isinstance(image_param, torch.Tensor) and image_param.device.type == 'cuda':
        image = image_param.cpu().numpy()
    elif isinstance(image_param, str):
        image = cv2.imread(image_param)
    elif isinstance(image_param, np.ndarray):
        image = image_param 
    else:
        image = cv2.imread('src/Temp_dir/img0.png')

    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    dlimits = dtype_limits(image, clip_negative=False)
    limits = np.percentile(image, [lower_percentile, upper_percentile])
    ratio = (limits[1] - limits[0]) / (dlimits[1] - dlimits[0])

    return ratio



def blur_score(image_param):
    """Calculate the blur score of an image"""
    # Convert the image to a NumPy array if it's a CUDA tensor
    if isinstance(image_param, torch.Tensor) and image_param.device.type == 'cuda':
        image = image_param.cpu().numpy()
    elif isinstance(image_param, str):
        image = cv2.imread(image_param)
    elif isinstance(image_param, np.ndarray):
        image = image_param 
    else:
        file_bytes = np.asarray(bytearray(image_param.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        cv2.imwrite('src/Temp_dir/img0.png', image)

    if len(image.shape) == 3:
        # Convert color image to grayscale
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        image_gray = image

    # Calculate the Laplacian of the image
    laplacian = cv2.Laplacian(image_gray, cv2.CV_64F)

    # Calculate the variance of the Laplacian as the blur score
    blur_score = laplacian.var()

    return blur_score



def sharpness_gradient_score(image_param):
    """Calculate the sharpness gradient score of an image"""
    # Convert the image to a NumPy array if it's a CUDA tensor
    if isinstance(image_param, torch.Tensor) and image_param.device.type == 'cuda':
        image = image_param.cpu().numpy()
    elif isinstance(image_param, str):
        image = cv2.imread(image_param)
    elif isinstance(image_param, np.ndarray):
        image = image_param
    else:
        image = cv2.imread('src/Temp_dir/img0.png')

    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image

    sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)

    magnitude_gradient = np.sqrt(sobel_x**2 + sobel_y**2)
    sharpness_gradient = np.mean(magnitude_gradient)

    return sharpness_gradient



def brightness_score(image_param):
    """Calculate the brightness score of an image"""
        # Convert the image to a NumPy array if it's a CUDA tensor
    if isinstance(image_param, torch.Tensor) and image_param.device.type == 'cuda':
        image = image_param.cpu().numpy()
    elif isinstance(image_param, str):
        image = cv2.imread(image_param)
    elif isinstance(image_param, np.ndarray):
        image = image_param 
    else:
        image = cv2.imread('src/Temp_dir/img0.png')

    levels = np.linspace(0, 255, num=1000)
    blue_channel, green_channel, red_channel = cv2.split(image)
    red_channel_mean = np.array(red_channel).mean()
    green_channel_mean = np.array(green_channel).mean()
    blue_channel_mean = np.array(blue_channel).mean()

    image_bright_value = np.sqrt(0.299 * (red_channel_mean ** 2) + 0.587 * (green_channel_mean ** 2) + 0.114 * (blue_channel_mean ** 2))

    image_bright_level = np.digitize(image_bright_value, levels, right=True) / 1000
    return image_bright_level

#############################################################################################################################


# Initialize the IQA metrics on GPU
maniqa = pyiqa.create_metric('maniqa')
niqe = pyiqa.create_metric('niqe')
ilniqe = pyiqa.create_metric('ilniqe')


def calculate_maniqa_score(image):
    """Calculate the ManiQA score for the given image on CPU"""
    # Convert the matrix (list of lists) to a NumPy array directly on CPU
    image_array = image.clone().detach().cpu().numpy()
    file_path = 'src/Temp_dir/img1.png'

    # Save the image to the Temporary directory path
    cv2.imwrite(file_path, image_array)
    maniqa_score = maniqa(file_path)

    if os.path.exists(file_path):
        print('IMAGE EXISTS in MANIQA on the following path: ',file_path)
        os.remove(file_path)
    else:
        print('\n IMAGE DOESNT EXIST IN MANIQA')

    return maniqa_score.item()



def calculate_niqe_score(image):
    """Calculate the NIQE score for the given image on CPU"""
    # Convert the matrix (list of lists) to a NumPy array directly on CPU
    image_array = image.clone().detach().cpu().numpy()
    file_path = 'src/Temp_dir/img2.png'

    # Save the image to the specified file path
    cv2.imwrite(file_path, image_array)

    # Save the image to the Temporary directory path
    niqe_score = niqe(file_path)

    if os.path.exists(file_path):
        print('IMAGE EXISTS in NIQE on the following path: ',file_path)
        os.remove(file_path)
    else:
        print('\n IMAGE DOESNT EXIST IN NIQE')

    return niqe_score.item()


def calculate_ilniqe_score(image):
    """Calculate the IL-NIQE score for the given image on CPU"""
    # Convert the matrix (list of lists) to a NumPy array directly on CPU
    image_array = image.clone().detach().cpu().numpy()
    file_path = 'src/Temp_dir/img3.png'

    # Save the image to the specified file path
    cv2.imwrite(file_path, image_array)
    ilniqe_score = ilniqe(file_path)

    if os.path.exists(file_path):
        print('IMAGE EXISTS in ILNIQE on the following path: ',file_path)
        os.remove(file_path)
    else:
        print('\n IMAGE DOESNT EXIST IN ILNIQE')

    return ilniqe_score.item()



def calculate_topiq_face_score(image):
    """Calculate the TOPIQ-NR-FACE score for the given image on CPU"""
    # Convert the matrix (list of lists) to a NumPy array directly on CPU
    topiq_face = pyiqa.create_metric('topiq_nr-face')

    image_array = image.clone().detach().cpu().numpy()
    file_path = 'src/Temp_dir/img4.png'

    # Save the image to the temporary file path
    cv2.imwrite(file_path, image_array)
    topiq_face_score = topiq_face(file_path)

    if os.path.exists(file_path):
        print('IMAGE EXISTS in TOPIQ FACE on the following path: ',file_path)
        os.remove(file_path)
    else:
        print('\n IMAGE DOESNT EXIST IN TOPIQ FACE')

    return topiq_face_score.item()

