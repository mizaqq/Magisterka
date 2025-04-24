import cv2
import numpy as np
from skimage.filters import threshold_sauvola
from pathlib import Path
import pytesseract
from pytesseract import Output
import re
import string


def grayscale(img, code=cv2.COLOR_BGR2GRAY):
    return cv2.cvtColor(img, code)


def otsu_threshold(img, thresh_val=0, max_val=255, thresh_type=cv2.THRESH_BINARY + cv2.THRESH_OTSU):
    _, thresh = cv2.threshold(img, thresh_val, max_val, thresh_type)
    return thresh


def adaptive_threshold(
    img,
    max_val=255,
    adaptive_method=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    threshold_type=cv2.THRESH_BINARY,
    block_size=11,
    C=2,
):
    return cv2.adaptiveThreshold(img, max_val, adaptive_method, threshold_type, block_size, C)


def sauvola_threshold(img, window_size=25):
    thresh_sauvola = threshold_sauvola(img, window_size=window_size)
    return (img > thresh_sauvola).astype(np.uint8) * 255


def gaussian_blur(img, kernel_size=(5, 5), sigma=0):
    return cv2.GaussianBlur(img, kernel_size, sigma)


def gaussian_blur1(img, kernel_size=(3, 3), sigma=0):
    return cv2.GaussianBlur(img, kernel_size, sigma)


def gaussian_blur2(img, kernel_size=(1, 1), sigma=0):
    return cv2.GaussianBlur(img, kernel_size, sigma)


def median_blur(img, ksize=3):
    return cv2.medianBlur(img, ksize)


def sharpen(img, kernel=None):
    if kernel is None:
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv2.filter2D(img, -1, kernel)


def resize(img, scale=2, interpolation=cv2.INTER_CUBIC):
    return cv2.resize(img, None, fx=scale, fy=scale, interpolation=interpolation)


def contrast_stretching(img, lower_percentile=2, upper_percentile=98):
    p_low, p_high = np.percentile(img, (lower_percentile, upper_percentile))
    return cv2.normalize(img, None, p_low, p_high, cv2.NORM_MINMAX)


def histogram_equalization(img):
    return cv2.equalizeHist(img)


def clahe(img, clipLimit=2.0, tileGridSize=(8, 8)):
    clahe_obj = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    return clahe_obj.apply(img)


def morphological_closing(img, kernel_size=(2, 2)):
    kernel = np.ones(kernel_size, np.uint8)
    return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)


def morphological_opening(img, kernel_size=(2, 2)):
    kernel = np.ones(kernel_size, np.uint8)
    return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)


def dilation(img, kernel_size=(2, 2), iterations=1):
    kernel = np.ones(kernel_size, np.uint8)
    return cv2.dilate(img, kernel, iterations=iterations)


def erosion(img, kernel_size=(2, 2), iterations=1):
    kernel = np.ones(kernel_size, np.uint8)
    return cv2.erode(img, kernel, iterations=iterations)


def bilateral_filter(img, d=9, sigmaColor=75, sigmaSpace=75):
    return cv2.bilateralFilter(img, d, sigmaColor, sigmaSpace)


def unsharp_mask(img, kernel_size=(3, 3), sigma=1, weight_img=1.5, weight_blurred=-0.5, gamma=0):
    blurred = cv2.GaussianBlur(img, kernel_size, sigma)
    return cv2.addWeighted(img, weight_img, blurred, weight_blurred, gamma)


def preprocess_text(text):
    text = re.compile('<.*?>').sub('', text)
    text = re.compile('[%s]' % re.escape(string.punctuation)).sub(' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\[[0-9]*\]', ' ', text)
    text = re.sub(r'[^\w\s]', '', text.strip())
    text = re.sub(r'\d', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\s*\b[a-zA-Z]\b\s*', ' ', text).strip()
    return text


def preprocess_cost(text: str):
    text = text.lower()
    text = text.strip()
    text = text.replace(',', '.')
    text = ''.join([char for char in text if not char.isalpha() or " "])
    pattern = re.compile(r'\d{1,3}\.\d{2}')
    return pattern.findall(text)
