import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
img = cv.imread('../butterfly.png', 1)
edges = cv.Canny(img,100,200)

rgb = cv.cvtColor(edges, cv.COLOR_GRAY2RGB) # RGB for matplotlib, BGR for imshow() !
# step 2: now all edges are white (255,255,255). to make it red, multiply with another array:
rgb *= np.array((1,0,0),np.uint8) # set g and b to 0, leaves red :)

 # step 3: compose:
out = np.bitwise_or(img, rgb)

plt.subplot(121)
plt.imshow(img,cmap = 'gray')
plt.title('Original Image')
plt.subplot(122)
plt.imshow(out,cmap = 'gray')
plt.title('Edge Image')


plt.show()