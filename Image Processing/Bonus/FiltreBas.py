
from tkinter import *

import cv2
from PIL import Image
from PIL import ImageTk
import sys
from tkinter import filedialog
import numpy as np



class FiltreBas:
    def __init__(self):
        print("salam")
    def print(self):
        print("momo")

    def FGaussien(self,img):
        newImg = cv2.GaussianBlur(img, (5, 5), 3)
        return newImg

    def gradient(self,img):
        img = cv2.Sobel(img, cv2.CV_8U, 1, 1, ksize=1)
        return img

    def laplacian(self,img):
        laplacian = cv2.Laplacian(img, cv2.CV_64F)
        return laplacian

