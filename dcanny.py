#!/usr/bin/python
import cv2
import cv
import numpy as np

mlowThreshold = 0
mmaxThreshold = 100
def CannyThreshold(lowThreshold):
    global mmaxThreshold
    mlowThreshold=lowThreshold   
    mmaxThreshold = cv2.getTrackbarPos('Max threshold', 'canny demo')
    detected_edges = cv2.GaussianBlur(gray,(3,3),0)
    detected_edges = cv2.Canny(detected_edges,mlowThreshold,mmaxThreshold,apertureSize = kernel_size)
    print mlowThreshold,mmaxThreshold
    dst = cv2.bitwise_and(img,img,mask = detected_edges)  # just add some colours to edges from original image.
    cv2.imshow('canny demo',dst)

def CannyMaxThreshold(maxThreshold):
    global mlowThreshold
    mmaxThreshold = maxThreshold
    mlowThreshold = cv2.getTrackbarPos('Min threshold', 'canny demo')
    detected_edges = cv2.GaussianBlur(gray,(3,3),0)
    detected_edges = cv2.Canny(detected_edges,mlowThreshold,mmaxThreshold,apertureSize = kernel_size)
    print mlowThreshold,maxThreshold
    dst = cv2.bitwise_and(img,img,mask = detected_edges)  # just add some colours to edges from original image.
    cv2.imshow('canny demo',dst)

ratio = 3
kernel_size = 3

img = cv2.imread('jusongguangyun1-1.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

cv2.namedWindow('canny demo', cv.CV_WINDOW_AUTOSIZE)

cv2.createTrackbar('Min threshold','canny demo', mlowThreshold, 500, CannyThreshold)
cv2.createTrackbar('Max threshold','canny demo',mmaxThreshold, 800, CannyMaxThreshold)

CannyThreshold(0)  # initialization
if cv2.waitKey(0) == 27:
    cv2.destroyAllWindows()
