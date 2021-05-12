from tkinter import *

import cv2
from PIL import Image
from PIL import ImageTk
from tkinter import filedialog
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
from Susan import Susan
from Fourier import Fourier


class MainWindow():
    def __init__(self, main):
        # first Frame
        main.config(background="#F3C007")
        main.geometry('700x600')
        self.menubar = Menu(main)
        main.config(menu=self.menubar)
        frame_left = Frame(width=100, height=300, relief=SUNKEN)
        frame_right = Frame()

        # canvas for image
        self.input_left = Label(frame_left, text="input image", font=("Helvetica", 12))
        self.canvas = Canvas(frame_left, width=500, height=400)
        self.canvas.pack()
        self.image = PhotoImage(file="")

        # set first image on canvas
        self.image_on_canvas = self.canvas.create_image((500 / 2), (400 / 2), anchor=CENTER, image=self.image)

        # button to change image right
        self.button = Button(frame_left, text="Choose ...", height=2, width=40, bg="#F3C007", command=self.onButton)
        self.button.pack()
        self.input_left.place(x=20, y=20)
        self.input_left.pack()

        # seconde frame
        # canvas for image
        self.input_right = Label(frame_right, text="output image", font=("Helvetica", 12))
        self.label_right = Label(frame_right)
        self.canvas_right = Canvas(frame_right, width=500, height=400)
        self.canvas_right.grid(row=0, column=3)
        self.input_right.grid(row=1, column=3)
        self.image_right = PhotoImage(file="")

        # set first image on canvas
        self.image_on_canvas_right = self.canvas_right.create_image((500 / 2), (400 / 2), anchor=CENTER, image=self.image_right)
        frame_left.pack(side=LEFT, padx=100, pady=10)
        frame_right.pack(side=RIGHT, padx=50, pady=10)

        # file path
        self.file_path = ""

        ## filtre pass haut or hight filter
        self.menuFiltreHaut()

        ## low filter menu
        self.passBas()

        ## bruit menu or blur menu 
        self.bruitMenu()

        ## operation elementaire or additional operation
        self.transElem()

        ## morph Maths
        self.morphologyMaths()

        ## susan
        self.susan()

        ## Fast Fourier Transform FFT
        self.fft()

    def onButton(self):
        self.callToSelect()
        self.canvas.itemconfig(self.image_on_canvas, image=self.image)

    # read image and return it
    def read(self):
        path = self.file_path
        imgcv = cv2.imread(path, 1)
        imgcv = cv2.cvtColor(imgcv, cv2.COLOR_BGR2GRAY)
        height, width = 400, 450
        imgcvr = cv2.resize(imgcv, (round(width), round(height)), interpolation=cv2.INTER_AREA)
        return imgcvr

    def saveToRight(self, imagecv):
        imagecv = cv2.resize(imagecv, (round(500), round(400)), interpolation=cv2.INTER_AREA)
        img = Image.fromarray(imagecv)
        img = ImageTk.PhotoImage(img)
        self.image_right = img
        self.canvas_right.itemconfig(self.image_on_canvas_right, image=self.image_right)

    def callToSelect(self):
        width = 450
        height = 400
        filepath = filedialog.askopenfilenames(
            title="Choose a file",
            filetypes=[
                ('image files', '.png')
            ])
        self.file_path = filepath [0]
        img_right = Image.open(filepath [0])
        img_right = img_right.resize((width, height), Image.ANTIALIAS)
        photoImg_right = ImageTk.PhotoImage(img_right)
        self.image = photoImg_right

        ### -------------------------------- high pass filtres------------------------------##

    def menuFiltreHaut(self):
        menuFiltre2 = Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="Filtre pass haut", menu=menuFiltre2)
        menuFiltre2.add_command(label="Gaussien", command=self.FGaussien)
        menuFiltre2.add_command(label="Gradient", command=self.gradient)
        menuFiltre2.add_command(label="Laplacian", command=self.laplacian)

    def FGaussien(self):
        img = self.read()
        kernel = np.array(
            [
                [-1, -1, -1],
                [-1, 8, -1],
                [-1, -1, -1]
            ])
        highpass_3x3 = ndimage.convolve(img, kernel)
        self.saveToRight(highpass_3x3)

    def gradient(self):
        img = self.read()
        newImg = cv2.Sobel(np.float32(img), cv2.CV_8U, 1, 1, ksize=1)
        self.saveToRight(newImg)

    def laplacian(self):
        img = self.read()
        laplacian = cv2.Laplacian(img, cv2.CV_64F)
        self.saveToRight(laplacian)

        ### -------------------------------- low pass filters------------------------------##
    def passBas(self):
        menuFiltre1 = Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="Filtre pass bas", menu=menuFiltre1)
        menuFiltre1.add_cascade(label="Bilateral Filter", command=self.bilaterale)
        menuFiltre1.add_command(label="Moyenneur", command=self.ImageMoyenne)
        menuFiltre1.add_command(label="Median", command=self.FiltreMedian)

    def FiltreMedian(self):
        img = self.read()
        print("img =", img [0] [0])
        temp = np.zeros(9)
        for i in range(1, img.shape [0] - 1):
            for j in range(1, img.shape [1] - 1):
                temp [0] = img [i - 1] [j - 1]
                temp [1] = img [i - 1] [j]
                temp [2] = img [i - 1] [j + 1]
                temp [3] = img [i] [j - 1]
                temp [4] = img [i] [j]
                temp [5] = img [i] [j + 1]
                temp [6] = img [i + 1] [j - 1]
                temp [7] = img [i + 1] [j]
                temp [8] = img [i + 1] [j + 1]
                img [i] [j] = self.trier(temp)
        self.saveToRight(img)

    def trier(self,t):
        for i in range(0, t.shape [0] - 1):
            for j in range(1, t.shape [0]):
                if (t [i] > t [j]):
                    f = t [i]
                    t [i] = t [j]
                    t [j] = f
        return t [4]

    # filter average
    def ImageMoyenne(self):
        img = self.read()
        newImg = cv2.blur(img, (3, 3))
        self.saveToRight(newImg)

    def bilaterale(self):
        img = self.read()
        blur = cv2.bilateralFilter(img, 9, 75, 75)
        self.saveToRight(blur)

    # ---------------------- some blur ----------------#
    def bruitMenu(self):
        menuBruit = Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="Bruit", menu=menuBruit)
        menuBruit.add_command(label="Poiver et sel ", command=self.poivreAndSel)
        menuBruit.add_command(label="Gaussien")

    def poivreAndSel(self):
        image = self.read()
        # row,col,ch = image.shape
        s_vs_p = 0.5
        amount = 0.04
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)  # prendre la partie entier
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
        out [coords] = 1

        # Pepper mode
        num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
        out [coords] = 0
        self.saveToRight(out)

    # ------------------------------- transformation elementary ----------------+----------#
    def transElem(self):
        menuTransformation = Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="Transformation élémentaire", menu=menuTransformation)
        menuTransformation.add_command(label="Niveau de grais", command=self.NiveauGray)
        menuTransformation.add_command(label="seuillage ", command=self.seuillage3d)
        menuTransformation.add_command(label="Miroir Vertical", command=self.inverseParColon)
        menuTransformation.add_command(label="Miroir horizontal", command=self.inverseParLinge)
        menuTransformation.add_command(label="Histogramme", command=self.hist)
        menuTransformation.add_command(label="conversion image", command=self.convertColor)

    # gray image
    def NiveauGray(self):
        image = self.read()
        self.saveToRight(image)

    def seuillage3d(self):
        img = self.read()
        img = cv2.split(img) [0]
        (retVal, newImg) = cv2.threshold(img, 147, 255, cv2.THRESH_BINARY)
        self.saveToRight(newImg)

    # mirror with column
    def inverseParColon(self):
        img = self.read()
        flipVertical = cv2.flip(img, 0)
        self.saveToRight(flipVertical)

    # mirror with line
    def inverseParLinge(self):
        img = self.read()
        img_rotate_90_counterclockwise = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        self.saveToRight(img_rotate_90_counterclockwise)

    def hist(self):
        img = cv2.imread(self.file_path, 1)
        color = ('b', 'g', 'r')
        for i, col in enumerate(color):
            histr = cv2.calcHist([img], [i], None, [256], [0, 256])
            plt.plot(histr, color=col)
            plt.xlim([0, 256])
        plt.savefig("toto.png")
        img = cv2.imread("toto.png")
        self.saveToRight(img)
        plt.close()

    def convertColor(self):
        img = self.read()
        convertedImage = 255 - img
        self.saveToRight(convertedImage)

    # ------------------------------------- morphology Mathematics ------------------------------#
    def morphologyMaths(self):
        Morphologie = Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="Morphologie mathématiques", menu=Morphologie)
        Morphologie.add_command(label="Erosion", command=self.erosion)
        Morphologie.add_command(label="Dilatation", command=self.delation)
        Morphologie.add_command(label="Ouverture", command=self.ouverture)
        Morphologie.add_command(label="Fermeture", command=self.fermeture)
        Morphologie.add_command(label="white top hat", command=self.whiteTopHat)
        Morphologie.add_command(label="Black top Hat", command=self.blackTopHat)
        Morphologie.add_command(label="Gradien Morphplogy", command=self.gradientMorph)
        Morphologie.add_command(label="Conteur interieur", command=self.contorIn)
        Morphologie.add_command(label="Conteur exterieur", command=self.contorEx)

    def erosion(self):
        img = self.read()
        kernel = np.ones((5, 5), np.uint8)
        img_erosion = cv2.erode(img, kernel, iterations=1)
        self.saveToRight(img_erosion)

    def delation(self):
        img = self.read()
        # Taking a matrix of size 5 as the kernel 
        kernel = np.ones((5, 5), np.uint8)
        img_dilation = cv2.dilate(img, kernel, iterations=1)
        self.saveToRight(img_dilation)

    # opening
    def ouverture(self):
        img = self.read()
        kernel = np.ones((5, 5), np.uint8)
        img_dilation = cv2.dilate(img, kernel, iterations=1)
        img_erosion = cv2.erode(img_dilation, kernel, iterations=1)
        self.saveToRight(img_erosion)

    # closing
    def fermeture(self):
        img = self.read()
        kernel = np.ones((5, 5), np.uint8)
        img_erosion = cv2.erode(img, kernel, iterations=1)
        img_dilation = cv2.dilate(img_erosion, kernel, iterations=1)
        self.saveToRight(img_dilation)

    def whiteTopHat(self):
        img = self.read()
        kernel = np.ones((5, 5), np.uint8)
        img_dilation = cv2.dilate(img, kernel, iterations=1)
        ouverture = cv2.erode(img_dilation, kernel, iterations=1)
        imageWhileTopHat = img - ouverture
        self.saveToRight(imageWhileTopHat)

    def blackTopHat(self):
        img = self.read()
        kernel = np.ones((5, 5), np.uint8)
        img_erosion = cv2.erode(img, kernel, iterations=1)
        fermeture = cv2.dilate(img_erosion, kernel, iterations=1)
        imageBlackTopHat = fermeture - img
        self.saveToRight(imageBlackTopHat)

    def gradientMorph(self):
        img = self.read()
        kernel = np.ones((5, 5), np.uint8)
        gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
        self.saveToRight(gradient)

    # inside contor
    def contorIn(self):
        img = self.read()
        kernel = np.ones((5, 5), np.uint8)
        img_erosion = cv2.erode(img, kernel, iterations=1)
        conteurIn = img - img_erosion
        self.saveToRight(conteurIn)

    # Outside contor
    def contorEx(self):
        img = self.read()
        kernel = np.ones((5, 5), np.uint8)
        img_dilation = cv2.dilate(img, kernel, iterations=1)
        conteurIn = img_dilation - img
        self.saveToRight(conteurIn)

    # ---------------------------------------detection of susan  ----------------------------#
    def susan(self):
        detection = Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="Detection edge", menu=detection)
        detection.add_command(label="Susan", command=self.susanFunction)
        detection.add_command(label="Hariss", command=self.harris)
        detection.add_command(label="Canny", command=self.canny)

    def susanFunction(self):
        s = Susan(self.file_path)
        img = s.call()
        self.saveToRight(img)

    def harris(self):
        filename = self.file_path
        img = cv2.imread(filename)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = np.float32(gray)
        dst = cv2.cornerHarris(gray, 2, 3, 0.04)
        # result is dilated for marking the corners, not important
        dst = cv2.dilate(dst, None)
        # Threshold for an optimal value, it may vary depending on the image.
        img [dst > 0.01 * dst.max()] = [0, 0, 255]
        self.saveToRight(img)

    def canny(self):
        img = self.read()
        edges = cv2.Canny(img, 100, 200)
        self.saveToRight(edges)

    # --------------------------------------TFF---------------------------------------------#
    def fft(self):
        fft = Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="FFT", menu=fft)
        fft.add_command(label="Magnitude Spectru", command=self.mangnitudeSpec)
        fft.add_command(label="High Pass Filter (HPF)", command=self.fftHighPass)
        fft.add_command(label="Low Pass Filter (LPF)", command=self.LowPass)
        fft.add_command(label="Band Pass Filter (BPF)", command=self.bondPass)

    def mangnitudeSpec(self):
        img = self.read()
        f = np.fft.fft2(img)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20 * np.log(np.abs(fshift))
        self.saveToRight(magnitude_spectrum)

    def fftHighPass(self):
        f = Fourier(self.file_path)
        f.highPass()

    def LowPass(self):
        f = Fourier(self.file_path)
        f.lowPass()

    def bondPass(self):
        f = Fourier(self.file_path)
        f.bondPass()

    # ----------------------------------------------------------------------


root = Tk()
MainWindow(root)
root.mainloop()
