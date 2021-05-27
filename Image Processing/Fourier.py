import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import Image as im
class Fourier():

    def __init__(self,path):
        file = path
        self.img = cv2.imread(file,0)

    ## fft transfer with shift
    def fft(self):
        dft = cv2.dft(np.float32(self.img), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)
        return dft_shift

    # magnitude spectrum
    def magnitudeSpec(self):
        dft_shift = self.fft()
        magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))
        return magnitude_spectrum

    # high pass filter
    def highPass(self):
        dft_shift = self.fft()
        rows, cols = self.img.shape
        crow, ccol = int(rows / 2), int(cols / 2)  # center
        # Circular HPF mask, center circle is 0, remaining all ones
        mask = np.ones((rows, cols, 2), np.uint8)
        r = 50
        center = [crow, ccol]
        x, y = np.ogrid [:rows, :cols]
        mask_area = (x - center [0]) ** 2 + (y - center [1]) ** 2 <= r * r
        mask [mask_area] = 0

        # apply mask and inverse DFT
        fshift = dft_shift * mask
        fshift_mask_mag = 2000 * np.log(cv2.magnitude(fshift[:, :, 0], fshift[:, :, 1]))
        f_ishift = np.fft.ifftshift(fshift)
        img_back = cv2.idft(f_ishift)
        img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
        self.showInPlot(fshift_mask_mag, img_back)

    # low pass filter
    def lowPass(self):
        dft_shift = self.fft()
        rows, cols = self.img.shape
        crow, ccol = int(rows / 2), int(cols / 2)  # center
        # Circular HPF mask, center circle is 0, remaining all ones
        mask = np.zeros((rows, cols, 2), np.uint8)
        r = 100
        center = [crow, ccol]
        x, y = np.ogrid [:rows, :cols]
        mask_area = (x - center [0]) ** 2 + (y - center [1]) ** 2 <= r * r
        mask [mask_area] = 1

        fshift = dft_shift * mask
        fshift_mask_mag = 2000 * np.log(cv2.magnitude(fshift[:, :, 0], fshift[:, :, 1]))
        f_ishift = np.fft.ifftshift(fshift)
        img_back = cv2.idft(f_ishift)
        img_back = cv2.magnitude(img_back [:, :, 0], img_back [:, :, 1])
        self.showInPlot(fshift_mask_mag,img_back)

    # bond pass filter
    def bondPass(self):
        dft_shift=self.fft()
        rows, cols = self.img.shape
        crow, ccol = int(rows / 2), int(cols / 2)

        mask = np.zeros((rows, cols, 2), np.uint8)
        r_out = 100
        r_in =  50
        center = [crow, ccol]
        x, y = np.ogrid [:rows, :cols]
        mask_area = np.logical_and(((x - center [0]) ** 2 + (y - center [1]) ** 2 >= r_in ** 2),
                                   ((x - center [0]) ** 2 + (y - center [1]) ** 2 <= r_out ** 2))
        mask [mask_area] = 1

        fshift = dft_shift * mask
        fshift_mask_mag = 2000 * np.log(cv2.magnitude(fshift [:, :, 0], fshift [:, :, 1]))
        f_ishift = np.fft.ifftshift(fshift)
        img_back = cv2.idft(f_ishift)
        img_back = cv2.magnitude(img_back [:, :, 0], img_back [:, :, 1])
        self.showInPlot(fshift_mask_mag,img_back)

    # show in 4 plot with details
    def showInPlot(self,fshift_mask_mag,img_back):
        
        plt.subplot(2, 2, 1), plt.imshow(self.img, cmap='gray')
        plt.title('Input Image'), plt.xticks([]), plt.yticks([])
        plt.subplot(2, 2, 2), plt.imshow(self.magnitudeSpec(), cmap='gray')
        plt.title('After FFT'), plt.xticks([]), plt.yticks([])
        plt.subplot(2, 2, 3), plt.imshow(fshift_mask_mag, cmap='gray')
        plt.title('FFT + Mask'), plt.xticks([]), plt.yticks([])
        plt.subplot(2, 2, 4), plt.imshow(img_back, cmap='gray')
        plt.title('After FFT Inverse'), plt.xticks([]), plt.yticks([])
        plt.show()
        plt.savefig("fft.png")
