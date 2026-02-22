import cv2 as cv

# Read in an image
img = cv.imread('owl.jpg')
cv.imshow('Owl', img)

# Converting to grayscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)

# Blur 
blur = cv.GaussianBlur(img, (7,7), cv.BORDER_DEFAULT)  # (7,7) this always has to be an odd number (its basically a window) also this is the thing ot control how much blur
cv.imshow('Blur', blur)

# Edge Cascade
canny = cv.Canny(blur, 125, 175)  #cv.Canny(img,threshold 1, threshold 2)
cv.imshow('Canny Edges', canny)  #shows only edges rest black
"""In OpenCV’s Canny edge detector, threshold1 (minVal) and threshold2 (maxVal) are used for hysteresis thresholding to 
   determine strong and weak edges. threshold2 (upper) marks pixels as certain edges (white), 
   while threshold1 (lower) filters out noise, keeping only weak edges that are connected to strong ones, 
   creating clean edge lines."""

 
# Dilating the image
dilated = cv.dilate(canny, (7,7), iterations=3)  #cv.dilate(canny image,kernel size,iterations)
cv.imshow('Dilated', dilated)
"""Dilating an image is a morphological image processing operation that expands the boundaries of 
   foreground objects (usually white pixels) by adding pixels to their edges, essentially making white features larger and thicker.
   It is used to fill in small holes, connect broken lines, and remove noise. """

# Eroding
eroded = cv.erode(dilated, (7,7), iterations=3) #fix the dilated image (dilated image,kernel size,iterations)
cv.imshow('Eroded', eroded)

# Resize
resized = cv.resize(img, (500,500), interpolation=cv.INTER_CUBIC)  # cv.resize(img,resize dimensions, );  for resize to small size use INTER_AREA and for large size use INTER_CUBIC or INTER_LINEAR
cv.imshow('Resized', resized)  

# Cropping
cropped = img[50:200, 100:200]  # treats image like a matrix and specifies the exact indices(here the x, y coords) to be cropped; also if the indices are out of bounds then you will get error (-215:Assertion failed)
cv.imshow('Cropped', cropped)

cv.waitKey(0)