import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

class Susan():
    def __init__(self,img):
        self.img = img

    def call(self):
        img = cv.imread(self.img, 0)
        output1 = self.susan_corner_detection(img)
        finaloutput1 = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
        finaloutput1 [output1 != 0] = [255, 255, 0]
        return finaloutput1
    def plot_image(self,image,title):
        plt.figure()
        plt.title(title)
        plt.imshow(image,cmap = 'gray')
        plt.show()
        plt.savefig("toto")

    #susam mask of 37 pixels
    def susan_mask(self):
        mask=np.ones((7,7))
        mask[0,0]=0
        mask[0,1]=0
        mask[0,5]=0
        mask[0,6]=0
        mask[1,0]=0
        mask[1,6]=0
        mask[5,0]=0
        mask[5,6]=0
        mask[6,0]=0
        mask[6,1]=0
        mask[6,5]=0
        mask[6,6]=0
        return mask

    def susan_corner_detection(self,img):
        img = img.astype(np.float64)
        g=37/2
        circularMask=self.susan_mask()
        #print circularMask
        # img=create10by10Mask()
        # print(img)
        output=np.zeros(img.shape)
        #val=np.ones((7,7))

        for i in range(3,img.shape[0]-3):
            for j in range(3,img.shape[1]-3):
                ir=np.array(img[i-3:i+4, j-3:j+4])
                ir =  ir[circularMask==1]
                ir0 = img[i,j]
                a=np.sum(np.exp(-((ir-ir0)/10)**6))
                if a<=g:
                    a=g-a
                else:
                    a=0
                output[i,j]=a
        return output