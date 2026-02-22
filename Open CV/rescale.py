import cv2 as cv

img= cv.imread('owl.jpg')
cv.imshow('Owl',img)

def rescaleFrame(frame,scale=0.75):
    #images, videos and live_videos
    width=int(frame.shape[1] * scale) # here shape[1] represents width and shape[0] represents height
    height =int(frame.shape[0] * scale) # and frame.shape[]* scale is a float value, so we type cast to int
    dimensions=(width,height)
    #type cast to int in python simply ignores the digits the after decimal point and returns only integers

    return cv.resize(frame,dimensions,interpolation=cv.INTER_AREA)

"""Interpolation is the process of using known data to estimate values at unknown locations. This works in two directions and tries to achieve the best approximation of a pixel’s intensity based on the values of surrounding pixels. As it’s an approximation method image will always lose some quality when interpolated.

Image interpolation occurs especially when an image is resized or distorted(remapped) from a one-pixel grid to another."""

def chageRes(width,height):
    # Live Video
    capture.set(3,width) # Here 3 means width
    capture.set(4,width) # Here 4 means height

image_resized=rescaleFrame(img,scale=1)
cv.imshow('Resized_Owl',image_resized)

# capture=cv.VideoCapture('Path.mp4') 

# while True:
#     isTrue,frame = capture.read() 

#     resized_frame= rescaleFrame(frame)

#     cv.imshow('Video',frame)  
#     cv.imshow('Video_Resized',resized_frame)  

#     if cv.waitKey(20) & 0xFF==ord('d') :  
#         break

# capture.release()
# cv.destroyAllWindows()

cv.waitKey(0) 