import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.color import rgb2hsv, hsv2rgb
from skimage.filters.rank import median

# -- Your code here -- #
im = imread("resources/etretat.jpg")
im_hsv = rgb2hsv(im)


def gamma_lut(gamma, nb_levels):
    return nb_levels ** (1 - gamma) * (np.arange(nb_levels + 1) ** gamma)
# Saturation and value improvement using gamma lut


levels = 1000
lut_gamma = gamma_lut(0.5, levels)

sat_vals = (im_hsv[:, :, 1]*levels).astype(int)
sat_vals = lut_gamma[sat_vals] / levels
im_hsv[:, :, 1] = sat_vals


lut_gamma = gamma_lut(1.6, levels)
values = (im_hsv[:, :, 2]*levels).astype(int)
values = lut_gamma[values] / levels
im_hsv[:, :, 2] = values


k_size = 2
median_kern = np.ones((k_size, k_size))

# Smoothing color distribution using median filter on the hue
med_filt = median(im_hsv[:, :, 0], median_kern)
med_filt = med_filt / med_filt.max()
im_hsv[:, :, 0] = med_filt

im_better = hsv2rgb(im_hsv)


plt.figure(figsize=[12, 5])
plt.subplot(1, 2, 1)
plt.imshow(im)
plt.subplot(1, 2, 2)
plt.imshow(im_better)
plt.show()

