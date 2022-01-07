from skimage.io import imread, imsave
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2hsv, hsv2rgb

wat_mark = imread("resources/watermark.png")
im = imread('resources/etretat.jpg')

## Parameters
a = 0.4
position = (400, 650)
y,x = position

mask = wat_mark != 0
big_mask = np.zeros((im.shape[0], im.shape[1]))!=0
big_mask[x:x+mask.shape[0],y:y+mask.shape[1]] = mask

result = rgb2hsv(im)

values = result[:,:,2]

## Compute if the avg value of the mask is light or dark
avg = values[big_mask >0].mean()

coef = 1
if avg>0.5:
    coef = -1
a *= coef


values[big_mask] += a*values[big_mask]
values[values >1] = 1
values[values < 0] = 0


result[:,:,2] = values
im = hsv2rgb(result)

imsave("results/wat_im.jpg", im)
fig = plt.figure()
plt.imshow(im)
plt.colorbar()
plt.show()
