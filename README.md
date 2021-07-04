# imageProcessing

The image processing using the skimage and scipy library.

Q2.1

The image was read using imageio and .shape was used to determine the size of the image. To convert the image to 
binary, rgb2gray was used and the otsu threshold method was used to convert the image into binary.

Q2.2 

skimage was used to add random noise to the image. The image was filtered with a gaussian mask and then a
uniform smoothing mask with size (9x9) was used. 

Q2.3 

The image was used kmeans segmentation using the segmentation.slic and the boundaries were marked to show the 
different boundaries in the original image.

Q2.4 

Canny Edge detection was performed by converting the image into grayscale and the using the skimage library 
to perform canny edge detection. A sigma value of 1.18 was used after trial and error. A higher sigma value showed 
fewer edges on the image while a lower sigma showed more lines. 

Hough line transformation was performed on the canny image using the skimage examples. 
