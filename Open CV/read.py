import cv2 as cv

""" reading images
img= cv.imread('owl.jpg')

cv.imshow('Cat',img)

cv.waitKey(0)       """

#Reading videos
#in video we read the video frame by frame  by using a while True loop
capture=cv.VideoCapture(0) #this is used with 0 when referncing camera devices; 0 is for webcam
capture=cv.VideoCapture('Path.mp4') #path is used for referecing the path of the video

while True:
    isTrue,frame = capture.read() # reads video frame by frame, it returns a boolean value which says whether the frame was successfully read or not and 
                                  # also returns the frame. 

    cv.imshow('Video',frame)  #for dispalying each frame 

    if cv.waitKey(20) & 0xFF==ord('d') :  #for stopping the video to play indefinetly if the letter 'd' is pressed 
        break
#if -215:Assertion failed error comes, that means that open cv could not find more frames in specified file path, this also happens with images if wrong path is specified

capture.release()
cv.destroyAllWindows()
