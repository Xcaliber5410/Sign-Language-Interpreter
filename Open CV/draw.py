import cv2 as cv
import numpy as np


blank = np.zeros((500,500,3),dtype='uint8') # this creates a blank image (500,500,3)(height,width,no. of color channels), dtype=data type
cv.imshow('Blank',blank)    #color combo is bgr not rgb

# 1. Paint the image a certain colour
#if you want to paint entire image then blank[:]
blank[200:300, 300:400] = 0,255,0  # (200:300, 300:400) paints a certain portion of the image 
cv.imshow('Green', blank)

# 2. Draw a Rectangle
cv.rectangle(blank,(0,0),(100,250),(0,255,0),thickness=-1)
cv.rectangle(blank, (0,0), (blank.shape[1]//2, blank.shape[0]//2), (0,255,0), thickness= 2)  # cv.rectangle(image, start point, end point, color, thickness)  
cv.imshow('Rectangle', blank)   #thickness =CV.FILLED , which will fill the entire rectangle or use -1
#blank.shape[1]//2, blank.shape[0]//2   this basically means half the image height and width

# 3. Draw A circle
cv.circle(blank, (blank.shape[1]//2, blank.shape[0]//2), 40, (0,0,255), thickness=-1) #cv.circle(img,centre,radius,color,thickness,linetype(additional))
cv.imshow('Circle', blank)

# 4. Draw a line
cv.line(blank, (100,250), (300,400), (255,255,255), thickness=3) #cv.line(img,start point, end point,color,thickness,linetype)
cv.imshow('Line', blank)

# 5. Write text
cv.putText(blank, 'Hello, my name is Jason!!!', (0,225), cv.FONT_HERSHEY_SCRIPT_COMPLEX, 1.0, (255,255,255), 2) # cv.putText(img,text,start point,font style,font scale,color,thickness)
cv.imshow('Text', blank)

cv.waitKey(0)

cv.waitKey(0)
