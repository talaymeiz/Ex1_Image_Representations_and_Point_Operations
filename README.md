# Ex1_Image_Representations_and_Point_Operations
# IMAGE PROCESSING
## Exercise 1 : Image Representations and Point Operations

Python 3.9.7 , PyCharm<br>
### ex1_utils.py:<br>
this function returns the normalized image with the requested color space representation.<br>
<div dir='ltr'>
  
    def imReadAndConvert(filename: str, representation: int) -> np.ndarray:
    """
    Reads an image, and returns the image converted as requested
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: The image object
    """
  
</div>

this function shows the image with the requested color space representation.<br>
<div dir='ltr'>
  
    def imDisplay(filename: str, representation: int):
    """
    Reads an image as RGB or GRAY_SCALE and displays it
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: None
    """
  
</div>

this function transforms the image, from RGB to YIQ color space representation.<br>
<div dir='ltr'>
  
    def transformRGB2YIQ(imgRGB: np.ndarray) -> np.ndarray:
    """
    Converts an RGB image to YIQ color space
    :param imgRGB: An Image in RGB
    :return: A YIQ in image color space
    """
  
</div>

this function transforms the image, from YIQ to RGB color space representation.<br>
<div dir='ltr'>
  
    def transformYIQ2RGB(imgYIQ: np.ndarray) -> np.ndarray:
    """
    Converts an YIQ image to RGB color space
    :param imgYIQ: An Image in YIQ
    :return: A RGB in image color space
    """
  
</div>

this function perform histogram equalization of a given grayscale or RGB image.<br>
<div dir='ltr'>
  
    def hsitogramEqualize(imgOrig: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
    """
    Equalizes the histogram of an image
    :param imgOrig: Original Histogram
    :ret
    """
  
</div>

this function perform optimal quantization of a given grayscale or RGB image.<br>
<div dir='ltr'>
  
    def quantizeImage(imOrig: np.ndarray, nQuant: int, nIter: int) -> (List[np.ndarray], List[float]):
    """
    Quantized an image in to *nQuant* colors
    :param imOrig: The original image (RGB or Gray scale)
    :param nQuant: Number of colors to quantize the image to
    :param nIter: Number of optimization loops
    :return: (List[qImage_i],List[error_i])
    """
  
</div>



### gamma.py:<br>
this function shows the the imge and the slide of gamma correction.<br>
<div dir='ltr'>
  
    def gammaDisplay(img_path: str, rep: int):
    """
    GUI for gamma correction
    :param img_path: Path to the image
    :param rep: grayscale(1) or RGB(2)
    :return: None
    """
  
</div>

this inside function makes the gamma correction.<br>
<div dir='ltr'>
  
    def gammaTrackbar(GammaValue):
  
</div>
