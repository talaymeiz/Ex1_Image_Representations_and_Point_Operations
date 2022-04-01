"""
        '########:'##::::'##::::'##:::
         ##.....::. ##::'##:::'####:::
         ##::::::::. ##'##::::.. ##:::
         ######:::::. ###::::::: ##:::
         ##...:::::: ## ##:::::: ##:::
         ##:::::::: ##:. ##::::: ##:::
         ########: ##:::. ##::'######:
        ........::..:::::..:::......::
"""
from typing import List
import cv2
import numpy as np
from matplotlib import pyplot as plt

LOAD_GRAY_SCALE = 1
LOAD_RGB = 2


def myID() -> np.int:
    """
    Return my ID (not the friend's ID I copied from)
    :return: int
    """
    return 207883430


def imReadAndConvert(filename: str, representation: int) -> np.ndarray:
    """
    Reads an image, and returns the image converted as requested
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: The image object
    """
    # src = cv2.imread(filename)
    #
    # if len(src)==2: #grey
    #     image = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    # else:                 #RGB
    #     image = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    #
    # norm_image = cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    #
    # return image
    img = cv2.imread(filename)
    if representation == 1:  # GRAY_SCALE
        img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        imageGray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imageGray = cv2.cvtColor(imageGray, cv2.COLOR_RGB2GRAY)
        return imageGray
    else:  # representation == 2 , RGB
        img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        imageRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return imageRGB


def imDisplay(filename: str, representation: int):
    """
    Reads an image as RGB or GRAY_SCALE and displays it
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: None
    """
    img = imReadAndConvert(filename, representation)
    plt.imshow(img)
    plt.show()



def transformRGB2YIQ(imgRGB: np.ndarray) -> np.ndarray:
    """
    Converts an RGB image to YIQ color space
    :param imgRGB: An Image in RGB
    :return: A YIQ in image color space
    """
    Y_from_RGB = np.array([[0.299,0.587,0.114],[0.596,-0.275,-0.321],[0.212,-0.523,0.311]])
    Y = imgRGB @ Y_from_RGB.transpose()
    return Y


def transformYIQ2RGB(imgYIQ: np.ndarray) -> np.ndarray:
    """
    Converts an YIQ image to RGB color space
    :param imgYIQ: An Image in YIQ
    :return: A RGB in image color space
    """
    Y_from_RGB = np.array([[0.299,0.587,0.114],[0.596,-0.275,-0.321],[0.212,-0.523,0.311]])
    R_from_Y = np.linalg.inv(Y_from_RGB)
    RGB = imgYIQ @ R_from_Y.transpose()
    return RGB


def hsitogramEqualize(imgOrig: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
    """
        Equalizes the histogram of an image
        :param imgOrig: Original Histogram
        :ret
    """
    n_imgOrig = cv2.normalize(imgOrig, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    n_imgOrig = n_imgOrig.astype(np.uint8)

    ans2histOrg = np.zeros(256)
    for val in range(256):
        ans2histOrg[val] = np.count_nonzero(n_imgOrig == val)

    cum_sum = np.cumsum(ans2histOrg)
    look_ut = np.floor((cum_sum / cum_sum.max()) * 255)

    ans1imgEq = np.zeros_like(imgOrig, dtype=float)
    for i in range(256):
        ans1imgEq[n_imgOrig == i] = int(look_ut[i])

    ans3histEQ = np.zeros(256)
    for val in range(256):
        ans3histEQ[val] = np.count_nonzero(ans1imgEq == val)

    # from range [0, 255] to range [0, 1]
    ans1imgEq = ans1imgEq / 255.0

    return ans1imgEq, ans2histOrg, ans3histEQ



# def quantizeImage(imOrig: np.ndarray, nQuant: int, nIter: int) -> (List[np.ndarray], List[float]):
#     """
#         Quantized an image in to **nQuant** colors
#         :param imOrig: The original image (RGB or Gray scale)
#         :param nQuant: Number of colors to quantize the image to
#         :param nIter: Number of optimization loops
#         :return: (List[qImage_i],List[MSE_error])
#     """
#
#     quantized_image= []
#     MSE_error= []
#
#     n_imgOrig = cv2.normalize(imOrig, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
#     n_imgOrig = n_imgOrig.astype(np.uint8)
#
#     histOrg = np.zeros(256)
#     for val in range(256):
#         histOrg[val] = np.count_nonzero(n_imgOrig == val)
#
#     # norm_image = cv2.normalize(imOrig, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
#     # flattenedorg = np.ndarray.flatten(norm_image)
#     # histOrg = np.histogram(flattenedorg)
#
#     z=  np.zeros(nQuant + 1, dtype=int)
#     for i in range(nQuant + 1):
#         z[i] = i * (255.0 / nQuant)
#
#     q= np.zeros(nQuant, dtype=int)
#
#     for i in range(nIter):
#         x_bar = []
#         for j in range(nQuant):
#             intense = histOrg[z[j]:z[j + 1]]
#             idx = range(len(intense))
#             weightedMean = (intense * idx).sum() / np.sum(intense)
#             x_bar.append(z[j] + weightedMean)
#
#         qImage_i = np.zeros_like(imOrig)
#
#         for k in range(len(x_bar)):
#             qImage_i[imOrig > z[k]] = x_bar[k]
#
#         mse = np.sqrt((imOrig - qImage_i) ** 2).mean()
#         MSE_error.append(mse)
#         quantized_image.append(qImage_i / 255.0)
#         for k in range(len(x_bar) - 1):
#             z[k + 1] = (x_bar[k] + x_bar[k + 1]) / 2
#
#     return quantized_image, MSE_error

def quantizeImage(imOrig: np.ndarray, nQuant: int, nIter: int) -> (List[np.ndarray], List[float]):
    """
        Quantized an image in to **nQuant** colors
        :param imOrig: The original image (RGB or Gray scale)
        :param nQuant: Number of colors to quantize the image to
        :param nIter: Number of optimization loops
        :return: (List[qImage_i],List[error_i])
    """

    #if RGB image
    isRGB = False
    if len(imOrig.shape) == 3:
        isRGB = True
        #transform to YIQ
        imgYIQ = transformRGB2YIQ(imOrig)
        #y-channel
        imOrig = imgYIQ[:, :, 0]

    imgOrigInt = (imOrig * 255).astype("uint8")
    #find original image histogram
    histOrig, _ = np.histogram(imgOrigInt.flatten(), 256, range=(0, 255))

    #mse array
    MSE_error = []
    #images list
    quantized_image = []
    global intensities, z, q

    for j in range(nIter):
        encodeImg = imgOrigInt.copy()
        #find z
        #initiate z
        if j == 0:
            z = np.arange(0, 255 - int(256 / nQuant) + 1, int(256 / nQuant))
            z = np.append(z, 255)
            intensities = np.array(range(256))
        else:
            for r in range(1, len(z) - 2):
                new_z_r = int((q[r - 1] + q[r]) / 2)
                if new_z_r != z[r - 1] and new_z_r != z[r + 1]:
                    z[r] = new_z_r

        #find q
        q = np.array([], dtype=np.float64)
        for i in range(len(z) - 1):
            mask = np.logical_and((z[i] < encodeImg), (encodeImg < z[i + 1]))
            if i is not (len(z) - 2):
                #calculate weighted mean
                if sum(histOrig[z[i]:z[i + 1]]) != 0:
                    q = np.append(q, np.average(intensities[z[i]:z[i + 1]], weights=histOrig[z[i]:z[i + 1]]))
                else:
                    q = np.append(q, np.average(intensities[z[i]:z[i + 1]], weights=histOrig[z[i]:z[i + 1]] + 0.001))
                encodeImg[mask] = int(q[i])

            else:
                #calculate weighted mean
                if sum(histOrig[z[i]:z[i + 1]]) != 0:
                    q = np.append(q, np.average(intensities[z[i]:z[i + 1] + 1], weights=histOrig[z[i]:z[i + 1] + 1]))
                else:
                    q = np.append(q, np.average(intensities[z[i]:z[i + 1] + 1],
                                                weights=histOrig[z[i]:z[i + 1] + 1] + 0.001))
                encodeImg[mask] = int(q[i])

        #find mse
        MSE_error.append((np.square(np.subtract(imgOrigInt, encodeImg))).mean())
        #normalize the image to [0,1]
        encodeImg = (encodeImg / 255)

        if isRGB:
            #y-channel
            imgYIQ[:, :, 0] = encodeImg.copy()
            #transform back to RGB
            encodeImg = transformYIQ2RGB(imgYIQ)

        quantized_image.append(encodeImg)

    return quantized_image, MSE_error


