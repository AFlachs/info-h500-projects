"""
We will do this by using two methods.
First, we use a watershed based segmentation and count the number of pixels labelled with the same value than
pixels in the tumor (which we need to identify manually).
Afterwards we use an histogram based segmentation (Otsu treshold after removing the black pixels from the histogram)
and compare the two values.
None of these methods are completely automated.
"""
from skimage.io import imread, imshow
import numpy as np
from skimage.morphology import disk
import skimage.filters.rank as skr
from skimage.segmentation import mark_boundaries, watershed
from skimage.feature import peak_local_max
from skimage.color import rgb2gray
from skimage.measure import label
import matplotlib.pyplot as plt
from matplotlib import cm

im_path = 'resources/mri_brain.jpg'

im = imread(im_path)
im = rgb2gray(im)

smoothing_factor = 3

# Compute the gradients of the image:
gradient = skr.gradient(skr.mean(im, disk(smoothing_factor)), disk(1))

# Defining markers
markers = np.zeros_like(im)

grad_i = gradient.max() - gradient  # no peak_local_min available so we inverse the grad image
markers_coords = peak_local_max(grad_i, min_distance=20, num_peaks=10)
markers[tuple(markers_coords.T)] = True
markers = label(markers)

ws = watershed(gradient, markers)

region_mean = np.zeros((ws.max() + 1, 1))
im_mean = np.zeros_like(im)

for i in range(ws.min(), ws.max() + 1):  # Smoothing regions based on their means
    region_mean[i] = im[ws == i].mean()
    im_mean[ws == i] = region_mean[i]

# We work on the part with the highest average value
mask = im_mean == im_mean.max()
nb_pixels = mask.sum()
print("Tumour size ws method :", nb_pixels * (0.115 ** 2), "cm^2")

plt.figure(figsize=[10, 10])
plt.subplot(2, 2, 1)
plt.imshow(im, cmap=plt.cm.gray)
plt.imshow(mask, alpha=0.25)
plt.plot(markers_coords[:, 1], markers_coords[:, 0], 'r+')
plt.subplot(2, 2, 2)
plt.imshow(im_mean, cmap=plt.cm.gray)
plt.subplot(2, 2, 3)
plt.imshow(ws)
plt.subplot(2, 2, 4)
plt.imshow(mark_boundaries(im, ws))
plt.show()

######################################################################
################### HISTOGRAM BASED METHOD ###########################
######################################################################


def comp_hist(image):
    hist, bins = np.histogram(image.flatten(), bins=range(257))
    return hist


def otsu_threshold(normed_hist):
    v = np.arange(256)

    bestT0 = 0
    bestl = 0
    for T in range(1, len(n_h)):
        w0 = n_h[:T].sum()
        w1 = n_h[T:].sum()

        m0 = (v[:T] * n_h[:T]).sum() / w0
        m1 = (v[T:] * n_h[T:]).sum() / w1

        s0 = (n_h[:T] * (v[:T] - m0) ** 2).sum() / w0
        s1 = (n_h[T:] * (v[T:] - m1) ** 2).sum() / w1

        sw = w1 * s1 + w0 * s0
        sb = w0 * w1 * ((m0 - m1) ** 2)
        l = sb / sw

        if l > bestl:
            bestT0 = T
            bestl = l

    return bestT0


# Using Otsu
im = imread(im_path)
im = rgb2gray(im) * 255
max_level = 70
# We remove low value from background
h = comp_hist(im)
new_hist = h
new_hist[:max_level] = 0
n_h = new_hist / new_hist.sum()

T = otsu_threshold(n_h)
im_segmented = im > T

plt.figure(figsize=(15, 5))
plt.plot(n_h)
plt.plot([T, T], [0, n_h.max()], 'r-')
plt.show()

# Show original image & segmented binary image
plt.figure(figsize=(15, 8))
plt.subplot(1, 2, 1)
plt.imshow(im, cmap=cm.gray)
plt.subplot(1, 2, 2)
plt.imshow(im_segmented, cmap=cm.gray)
plt.show()

nb_pixls = im_segmented.sum()
print("Estimated size :", nb_pixls * (0.115 ** 2), "cm^2")
print("The estimated size with the otsu TH is too high. This isn't a good method "
      "since it is only based on the color value and we know the tumor to be a connected surface.")
