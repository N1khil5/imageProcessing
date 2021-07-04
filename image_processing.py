import numpy as np
import pandas as pd
import sklearn
import scipy
import skimage
import imageio
import skimage.color as color
import matplotlib.pyplot as plt
import skimage.filters as filters
from skimage.filters import threshold_mean
from skimage.transform import hough_line, hough_line_peaks, probabilistic_hough_line
from skimage import segmentation
import skimage.util
import scipy.ndimage.filters as scifilters

# Q1 Converting the RGB image into Grayscale and Binary

avengers = imageio.imread("data/avengers_imdb.jpg")

avengersSize = avengers.shape

print("Size of Avengers image = ", avengersSize)

avengersGray = color.rgb2gray(avengers)

plt.imshow(avengersGray, cmap=plt.cm.gray)
plt.axis('off')
plt.tight_layout()
plt.savefig('outputs/avengersGray.jpg')
plt.show()


threshold = filters.threshold_otsu(avengersGray)
binary = avengersGray > threshold

plt.imshow(binary, cmap=plt.cm.gray, interpolation="nearest")
plt.axis('off')
plt.savefig('outputs/avengersBW.jpg')
plt.show()

# Q2

bushHouse = imageio.imread("data/bush_house_wikipedia.jpg")
randomNoise = skimage.util.random_noise(bushHouse, var=0.1)
plt.imshow(randomNoise)
plt.axis('off')
plt.savefig('outputs/bush_house_random_noise.jpg')
plt.show()

gMask = filters.gaussian(randomNoise, sigma=1)
plt.imshow(gMask)
plt.axis('off')
plt.savefig('outputs/bush_house_gaussian_mask.jpg')
plt.show()

uSmoothing = scifilters.uniform_filter(gMask, size=(9,9,1))
plt.imshow(uSmoothing)
plt.axis('off')
plt.savefig('outputs/bush_house_uniform_filter.jpg')
plt.show()

'''
Random Noise: https://scikit-image.org/docs/dev/api/skimage.util.html?highlight=random%20noise#skimage.util.random_noise
Gaussian Mask: https://scikit-image.org/docs/dev/api/skimage.filters.html?highlight=gauss#skimage.filters.gaussian
Uniform Smoothing Mask : https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.uniform_filter.html
'''

# Q3

forestCommission = imageio.imread("data/forestry_commission_gov_uk.jpg")
kmeans = skimage.segmentation.slic(forestCommission, n_segments=5, start_label=1)
#boundaries = skimage.segmentation.mark_boundaries(forestCommission, kmeans)
#plt.imshow(boundaries)
plt.imshow(kmeans)
plt.axis('off')
#plt.savefig('outputs/forestCommission_kmeans.jpg')
plt.savefig('outputs/forestCommission_kmeans_noBound.jpg')
plt.show()
'''
K-Means Segmentation: https://scikit-image.org/docs/dev/api/skimage.segmentation.html#skimage.segmentation.slic
'''

# Q4

rollandGaros = imageio.imread('data/rolland_garros_tv5monde.jpg')
rollandGarosGray = color.rgb2gray(rollandGaros)
cannyEdge = skimage.feature.canny(rollandGarosGray, sigma=1.18)
plt.imshow(cannyEdge, cmap=plt.cm.gray)
plt.axis('off')
plt.savefig('outputs/rollandGarros_cannyedge.jpg')
plt.show()

plt.imshow(cannyEdge, cmap=plt.cm.gray)
tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 360, endpoint=False)
h, theta, d = hough_line(cannyEdge, theta=tested_angles)

for _, angle, dist in zip(*hough_line_peaks(h, theta, d)):
    (x0, y0) = dist * np.array([np.cos(angle), np.sin(angle)])
    plt.axline((x0, y0), slope=np.tan(angle + np.pi/2))
plt.tight_layout()
plt.axis('off')
plt.savefig('outputs/rollandGarros_houghLine.jpg')
plt.show()

'''
References:
Canny Edge: https://scikit-image.org/docs/dev/auto_examples/edges/plot_canny.html
Hough Transform: https://scikit-image.org/docs/0.8.0/auto_examples/plot_hough_transform.html
'''