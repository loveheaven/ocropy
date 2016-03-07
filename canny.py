#!/usr/bin/python
import sys
import cv2
from cv import *
import random
import numpy as np
from scipy import ndimage
from matplotlib import pyplot as plt
import math
from pylab import *
import argparse,glob,os,os.path
import traceback
from scipy.ndimage import measurements,interpolation,morphology,filters
from scipy.misc import imsave
from scipy.ndimage.filters import gaussian_filter,uniform_filter,maximum_filter,rank_filter
from multiprocessing import Pool
import ocrolib
from ocrolib import psegutils,morph,sl
from ocrolib.toplevel import *
import os
import os.path

def caculateRect(box):
    left = -1
    top = -1
    bottom = -1
    right = -1
    for ar in box:
        if left > ar[0] or left == -1:
            left=ar[0]
        if right < ar[0]:
            right=ar[0]
        if top > ar[1] or top == -1:
            top=ar[1]
        if bottom < ar[1]:
            bottom=ar[1]
    return left,top,right,bottom

def find_border_components(contours, ary):
    borders = []
    area = ary.shape[0] * ary.shape[1]
    for i, c in enumerate(contours):
        x,y,w,h = cv2.boundingRect(c)
        if w * h > 0.1 * area:
            borders.append((i, x, y, x + w - 1, y + h - 1))
    return borders

def vprojection(img):
#    height, width = img.shape
#    projection = np.zeros(width)
#    for x in range(0, width):
#        for y in range(0, height):
#            if img[y,x] > 0:
#                print img[y,x]
#                projection[x] += 1
    projection = np.sum(img, axis=0)/255
    return projection

def hprojection(img):
#    height, width = img.shape
#    projection = np.zeros(height)
#    
#    for y in range(0, height):
#        for x in range(0, width):
#            if img[y,x] > 0:
#                print img[y,x]
#                projection[y] += 1
    projection = np.sum(img, axis=1)/255
#    for i in range(0, height):
#        if projection[i] != projection1[i]:
#            print "error",i,projection[i],projection1[i]
#            for x in range(0, width):
#                print img[i,x],img[i][x]
#            exit(1)
    return projection

def cropProjection(projection, threshold = 0):
    min = -1
    max = -1
    length = len(projection)
    for i in range(0, length):
        if projection[i] > threshold:
            min = i
            break
    for i in range(length-1, 0, -1):
        if projection[i] > threshold:
            max = i
            break
    return (min, max)

def splitProjection(projection, min, max, lengthThreshold=50, splitThreshold=0):
    ret = []
    start = min - 1
    length = 0
    for i in range(min, max):
        if projection[i] <= splitThreshold:
            if length >= lengthThreshold:
                ret.append((start+1, i - 1))
            start = i
            length = 0
        else:
            length+=1
    if length >= lengthThreshold:
        ret.append((start+1, max))
    return ret

def findMiddleRegion(projection, min, max):
    ret = []
    length = 0
    start = min - 1
    for i in range(min, max):
        if projection[i] == 0:
            length += 1
        else:
            if length >= 50:
                ret.append((start+1, i - 1))
            start = i
            length = 0
    if length >= 50:
        ret.append((start+1, max))
    return ret

def findMiddleRegion(regions, center, minDistance=30):
    ret = []
    lastEnd = -1
    #this loop is to find all the gaps of which distance is more than 30
    for region in regions:
        if lastEnd == -1:
            lastEnd = region[1]
        else:
            if (region[0] - lastEnd) >= minDistance:
                ret.append((lastEnd, region[0]))
            lastEnd = region[1]
    min = 100000
    result = (-1,-1)
    #this loop is to find the closest gap with center
    for region in ret:
        if abs(region[0] - center) < min:
            min = abs(region[0] - center)
            result = region
    return result


def cropImage(edges, blur):
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#borders = find_border_components(contours, edges)
#borders.sort(key=lambda (i, x1, y1, x2, y2): (x2 - x1) * (y2 - y1))
#count=len(borders)
#print len(contours),count
#for i in range(0,count):
#    index, left,top,right,bottom =borders[i]
#    img_crop=cv2.getRectSubPix(img, (right-left, bottom-top), ((left+right)/2, (top+bottom)/2))
#    cv2.imwrite('1right%d' % (left) +'.png', img_crop)

#iand = cv2.bitwise_and(img,img,mask=edges)
#contours, hierarchy = cv2.findContours(edges,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#cv2.drawContours(edges,contours,-1,(255,255,255),-1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 50))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    #closed = cv2.erode(closed, None, iterations = 4)
    closed = cv2.dilate(closed, None, iterations = 13)
    #cv2.imwrite('closed.png',closed)
    (cnts, _) = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)
    for i in range(0,1):
        if len(cnts) == 0:
            break
        if i == 1 and len(cnts) == 1:
            break
        c=cnts[i]
        # compute the rotated bounding box of the largest contour
        rect = cv2.minAreaRect(c)
        box = np.int0(cv2.cv.BoxPoints(rect))
        #cv2.drawContours(img, [box], -1, (0, 255, 0), 3)
        left,top,right,bottom =caculateRect(box)
        img_crop=cv2.getRectSubPix(blur, (right-left, bottom-top), ((left+right)/2, (top+bottom)/2))
        cv2.imwrite('right%d' % (left) +'.png', img_crop)

def processImage(blur, lowthreshold=200, highThreshold=450, tag=''):
    edges = cv2.Canny(blur,lowthreshold,highThreshold)
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(edges,contours,-1,(255,255,255),-1)
    cv2.imwrite('edges%s.png' % tag,edges)
#    (_, img) = cv2.threshold(edges, 110, 255, cv2.THRESH_BINARY)
#    cv2.imwrite('edges3%s.png' % tag,img)
#    scale = psegutils.estimate_scale(img)
#    binary = remove_hlines(img, blur, scale)
#    cv2.imwrite('edges4%s.png' % tag,binary*255)
#    binary = remove_vlines(binary, blur, scale)
#    edges=binary*255
#    cv2.imwrite('edges2%s.png' % tag,edges)

#    minLineLength = 70
#    maxLineGap = 50
#    height, width = edges.shape
#    theta = 0
#    lines = cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength,maxLineGap)
#    if lines is not None:
#        for x1,y1,x2,y2 in lines[0]:
#            if x2 == x1:
#                cv2.line(edges,(x1,0),(x1,height),(0,0,0),2)
#                theta = 0
#            else:
#                tanTheta = float((y2-y1))/(x2-x1)
#                tmp = np.arctan2(y2-y1, x2-x1) * 180/np.pi
#                if tmp != 0.0 and tmp != 90.0:
#                    theta=math.fabs(tmp)
#                else:
#                    theta = 0
#                print theta, y2-y1, x2-x1
#                ystart=int((0-x1)*tanTheta+y1)
#                yend = int((width-x1)*tanTheta + y1)
#                cv2.line(edges,(0,ystart),(width,yend),(0,0,0),2)
#        cv2.imwrite('houghlines5%s.png' % tag,edges)
#        if theta > 85 and theta < 95:
#            theta = 90 - theta
#        elif theta < 5:
#            theta = -theta
#        elif theta > 175:
#            theta = 180 - theta
#        print tag,theta
#        if theta != 0:
#            blur = rotate(blur, theta)
#            cv2.imwrite('rotate%s.png' % tag,blur)
#
#            edges = cv2.Canny(blur,200,450)
#            contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#            cv2.drawContours(edges,contours,-1,(255,255,255),1)
#            cv2.imwrite('edges%s.png' % tag,edges)
#
#            minLineLength = 70
#            maxLineGap = 50
#            height, width = edges.shape
#            theta = 0
#            lines = cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength,maxLineGap)
#            for x1,y1,x2,y2 in lines[0]:
#                if x2 == x1:
#                    cv2.line(edges,(x1,0),(x1,height),(0,0,0),2)
#                else:
#                    tanTheta = float((y2-y1))/(x2-x1)
#                    theta = np.arctan2(y2-y1, x2-x1) * 180/np.pi
#                    ystart=int((0-x1)*tanTheta+y1)
#                    yend = int((width-x1)*tanTheta + y1)
#                    cv2.line(edges,(0,ystart),(width,yend),(0,0,0),2)
#            cv2.imwrite('houghlines5%s.png' % tag,edges)

#    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
#    edges = cv2.erode(edges, kernel, iterations = 1)
#    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
#    edges = cv2.erode(edges, kernel, iterations = 1)

    maxed_rows = rank_filter(edges, -4, size=(1, 30))#vertical
    maxed_cols = rank_filter(edges, -4, size=(30, 1))
    debordered = np.minimum(np.minimum(edges, maxed_rows), maxed_cols)#
    edges = debordered
    cv2.imwrite('edges1%s.png' % tag,edges)
    return edges

def splitImage(edges, blur, tag='',margin=4):
    h_projection = hprojection(edges)
    v_projection = vprojection(edges)
    top, bottom = cropProjection(h_projection)
    left, right = cropProjection(v_projection)

    #plt.imshow(edges,cmap = 'gray')
    #plt.plot(range(0, len(vprojection)), vprojection, 'r')
    #plt.plot(hprojection, range(0, len(hprojection)), 'b')
    #plt.show()
    regions = splitProjection(v_projection, left, right)
    print tag, left, right,top,bottom
    #print regions
    #print v_projection[1270:1450]
    if len(tag) == 0:
        return regions,left, right,top,bottom
    for region in regions:
        left, leftEnd = region
        if (leftEnd - left) > 220 and (leftEnd-left) < 300:
            width = (leftEnd - left)/2
            cr_img =cv2.getRectSubPix(blur, (width+margin, bottom-top+margin), (width/2+left, (top+bottom)/2))
            cv2.imwrite('crop%s-%d.png' % (tag, left), cr_img)
            cr_img =cv2.getRectSubPix(blur, (width+margin, bottom-top+margin), (leftEnd-(width/2), (top+bottom)/2))
            cv2.imwrite('crop%s-%d.png' % (tag, left+width), cr_img)
        else:
            cr_img =cv2.getRectSubPix(blur, (leftEnd-left+margin, bottom-top+margin), ((leftEnd+left)/2, (top+bottom)/2))
            cv2.imwrite('crop%s-%d.png' % (tag, left), cr_img)
    return regions,left, right,top,bottom

def isPixelBlack(pixel):
    if len(pixel) != 3:
        return False
    if pixel[0] == 0 and pixel[1] == 0 and pixel[2] == 0:
        return True
    return False

def erodePixel(img, row, col, kernelX, kernelY):
    height, width, channels = img.shape
    xRange = kernelX / 2
    yRange = kernelY / 2
    for color in range(channels):
        max = -1
        for i in range(kernelX):
            for j in range(kernelY):
                if (row-yRange + j) >= height or (col-xRange+i) >= width or (row-yRange + j) < 0 or (col-xRange+i) < 0:
                    continue
                if max <= img[row-yRange + j][col-xRange+i][color]:
                    max = img[row-yRange + j][col-xRange+i][color]
        for i in range(kernelX):
            for j in range(kernelY):
                if (row-yRange + j) >= height or (col-xRange+i) >= width or (row-yRange + j) < 0 or (col-xRange+i) < 0:
                    continue
                img[row-yRange + j][col-xRange+i][color] = max


def dialtePixel(img, row, col, kernelX, kernelY):
    height, width, channels = img.shape
    xRange = kernelX / 2
    yRange = kernelY / 2
    for color in range(channels):
        for i in range(kernelX):
            for j in range(kernelY):
                if (row-yRange + j) >= height or (col-xRange+i) >= width or (row-yRange + j) < 0 or (col-xRange+i) < 0:
                    continue
                img[row-yRange + j][col-xRange+i][color] = 0

def erode(img, iterations = 1):
    height, width, channels = img.shape
    for iter in range(iterations):
        for i in range(0,height,11):
            for j in range(0,width,11):
                if isPixelBlack(img[i][j]):
                    type = random.randint(1,2)
                    if type == 1:
                        #erodePixel(img, i, j, 5, 5)
                        pass
                    else:
                        dialtePixel(img, i, j, 11, 11)

def remove_hlines(binary,gray,scale,maxsize=10):
    labels,_ = morph.label(binary)
    objects = morph.find_objects(labels)
    for i,b in enumerate(objects):
        if sl.width(b)>maxsize*scale:
            gray[b][labels[b]==i+1] = 140
            labels[b][labels[b]==i+1] = 0
    return array(labels!=0, 'B')

def remove_vlines(binary,gray,scale,maxsize=10):
    labels,_ = morph.label(binary)
    objects = morph.find_objects(labels)
    for i,b in enumerate(objects):
        if (sl.width(b)<=20 and sl.height(b)>200) or (sl.width(b)<=45 and sl.height(b)>500):
            gray[b][labels[b]==i+1] = 140
            #gray[:,b[1].start:b[1].stop]=140
            labels[b][labels[b]==i+1] = 0
    return array(labels!=0, 'B')

def isfloatarray(a):
    return a.dtype in [dtype('f'),dtype('float32'),dtype('float64')]

def read_image_gray(a,pageno=0):
    """Read an image and returns it as a floating point array.
        The optional page number allows images from files containing multiple
        images to be addressed.  Byte and short arrays are rescaled to
        the range 0...1 (unsigned) or -1...1 (signed)."""
    if a.dtype==dtype('uint8'):
        a = a/255.0
    if a.dtype==dtype('int8'):
        a = a/127.0
    elif a.dtype==dtype('uint16'):
        a = a/65536.0
    elif a.dtype==dtype('int16'):
        a = a/32767.0
    elif isfloatarray(a):
        pass
    else:
        raise OcropusException("unknown image type: "+a.dtype)
    if a.ndim==3:
        a = mean(a,2)
    return a
def shift_dft(src, dst=None):
    '''
        Rearrange the quadrants of Fourier image so that the origin is at
        the image center. Swaps quadrant 1 with 3, and 2 with 4.
        src and dst arrays must be equal size & type
        '''
    
    if dst is None:
        dst = np.empty(src.shape, src.dtype)
    elif src.shape != dst.shape:
        raise ValueError("src and dst must have equal sizes")
    elif src.dtype != dst.dtype:
        raise TypeError("src and dst must have equal types")
    
    if src is dst:
        ret = np.empty(src.shape, src.dtype)
    else:
        ret = dst
    
    h, w = src.shape[:2]

    cx1 = cx2 = w/2
    cy1 = cy2 = h/2
    
    # if the size is odd, then adjust the bottom/right quadrants
    if w % 2 != 0:
        cx2 += 1
    if h % 2 != 0:
        cy2 += 1
    
    # swap quadrants
    
    # swap q1 and q3
    ret[h-cy1:, w-cx1:] = src[0:cy1 , 0:cx1 ]   # q1 -> q3
    ret[0:cy2 , 0:cx2 ] = src[h-cy2:, w-cx2:]   # q3 -> q1
    
    # swap q2 and q4
    ret[0:cy2 , w-cx2:] = src[h-cy2:, 0:cx2 ]   # q2 -> q4
    ret[h-cy1:, 0:cx1 ] = src[0:cy1 , w-cx1:]   # q4 -> q2
    
    if src is dst:
        dst[:,:] = ret
    
    return dst

def dftSkew(im):
    
    # convert to grayscale
    #im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    h, w = im.shape[:2]

    realInput = im.astype(np.float64)
    
    # perform an optimally sized dft
    dft_M = cv2.getOptimalDFTSize(w)
    dft_N = cv2.getOptimalDFTSize(h)
    
    # copy A to dft_A and pad dft_A with zeros
    dft_A = np.zeros((dft_N, dft_M, 2), dtype=np.float64)
    dft_A[:h, :w, 0] = realInput
    
    # no need to pad bottom part of dft_A with zeros because of
    # use of nonzeroRows parameter in cv2.dft()
    cv2.dft(dft_A, dst=dft_A, nonzeroRows=h)
    
    cv2.imshow("win", im)
    
    # Split fourier into real and imaginary parts
    image_Re, image_Im = cv2.split(dft_A)
    
    # Compute the magnitude of the spectrum Mag = sqrt(Re^2 + Im^2)
    magnitude = cv2.sqrt(image_Re**2.0 + image_Im**2.0)
    
    # Compute log(1 + Mag)
    log_spectrum = cv2.log(1.0 + magnitude)
    
    # Rearrange the quadrants of Fourier image so that the origin is at
    # the image center
    shift_dft(log_spectrum, log_spectrum)
    
    # normalize and display the results as rgb
    cv2.normalize(log_spectrum, log_spectrum, 0.0, 1.0, cv2.NORM_MINMAX)
    magMat=log_spectrum*255
    magMat= np.uint8(np.around(magMat))
    cv2.imwrite('dft.png', magMat)
    rows = h
    cols = w
    #    //imwrite("imageText_mag.jpg",magImg);

#    //Turn into binary image
    (_, magImg)=cv2.threshold(magMat,160,255,cv2.THRESH_BINARY);
    cv2.imwrite('dft1.png', magImg)
#    //imwrite("imageText_bin.jpg",magImg);

#    //Find lines with Hough Transformation
    pi180 = np.pi/180;
    linImg = np.zeros(magImg.shape)
    lines = cv2.HoughLines(magImg,1,pi180,100,0,0);
    print lines
    for line in lines[0]:
        rho = line[0]
        theta = line[1];
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        pt1 =(int(x0 + 1000*(-b)),int(y0 + 1000*(a)))
        pt2 = (int(x0 - 1000*(-b)),int(y0 - 1000*(a)))
        cv2.line(linImg,pt1,pt2,(255),1);
    cv2.imwrite("dlines.png",linImg);
#    //imwrite("imageText_line.jpg",linImg);
#    if(lines.size() == 3){
#        cout << "found three angels:" << endl;
#            cout << lines[0][1]*180/CV_PI << endl << lines[1][1]*180/CV_PI << endl << lines[2][1]*180/CV_PI << endl << endl;
#    }

#    //Find the proper angel from the three found angels
    angel=0;
    piThresh = np.pi/90;
    pi2 = np.pi/2;
    for line in lines[0]:
        theta = line[1];
        if abs(theta) < piThresh or abs(theta-pi2) < piThresh:
            continue;
        else:
            angel = theta;
            break;
    
#    //Calculate the rotation angel
#    //The image has to be square,
#    //so that the rotation angel can be calculate right
    if angel<pi2:
        angel=angel
    else:
        angel = angel-np.pi

    if angel != pi2:
        angelT = rows*tan(angel)/cols;
        angel = np.arctan(angelT);
    angelD = angel*180/np.pi;


    #Rotate the image to recover
    rotMat = cv2.getRotationMatrix2D((cols/2, rows/2),angelD,1.0);
    dstImg = cv2.warpAffine(im,rotMat,(cols,rows));
    cv2.imwrite("dresult.png",dstImg);

def estimate_skew_angle(image,angles):
    estimates = []
    for a in angles:
        v = mean(interpolation.rotate(image,a,order=0,mode='constant'),axis=1)
        v = var(v)
        estimates.append((v,a))
    _,a = max(estimates)
    return a

def estimate_angle(raw, maxskew=2,skewsteps=8,perc=80,range=20,zoom=0.5,bignore=0.1):
    comment = ""
    rawF = read_image_gray(raw)
    # perform image normalization
    image = rawF-amin(rawF)
    if amax(image)==amin(image):
        print "# image is empty",fname
        return
    image /= amax(image)

    extreme = (sum(image<0.05)+sum(image>0.95))*1.0/prod(image.shape)
    if extreme>0.95:
        comment += " no-normalization"
        flat = image
    else:
        # check whether the image is already effectively binarized
        # if not, we need to flatten it by estimating the local whitelevel
        m = interpolation.zoom(image,zoom)
        m = filters.percentile_filter(m,perc,size=(range,2))
        m = filters.percentile_filter(m,perc,size=(2,range))
        m = interpolation.zoom(m,1.0/zoom)
        w,h = minimum(array(image.shape),array(m.shape))
        flat = clip(image[:w,:h]-m[:w,:h]+1,0,1)

    # estimate skew angle and rotate
    d0,d1 = flat.shape
    o0,o1 = int(bignore*d0),int(bignore*d1)
    flat = amax(flat)-flat
    flat -= amin(flat)
    est = flat[o0:d0-o0,o1:d1-o1]
    ma = maxskew
    ms = int(2*maxskew*skewsteps)
    angle = estimate_skew_angle(est,linspace(-ma,ma,ms+1))
    return angle

def combineBoxmap(binary):
    objects = psegutils.binary_objects(binary)
    bysize = objects
    #sorted(objects,key=sl.area)
    def x_overlaps(u,v):
        return u[1].start<v[1].stop and u[1].stop>v[1].start
    def above(u,v):
        return u[0].start<v[0].start
    def left_of(u,v):
        return u[1].stop<v[1].start
    def separates(w,u,v):
        if w[0].stop<min(u[0].start,v[0].start): return 0
        if w[0].start>max(u[0].stop,v[0].stop): return 0
        if w[1].start<u[1].stop and w[1].stop>v[1].start: return 1
    order = zeros((len(bysize),len(bysize)),'B')
    for i,u in enumerate(bysize):
        #y1,y2,x1,x2
        #print u,u[0].start,u[0].stop,u[1].start, u[1].stop
        #binary[u]=0
#        M=cv2.moments(u)
#        x = int(M['m10']/M['m00'])
#        y = int(M['m01']/M['m00'])
        cv2.line(binary,(u[1].start, u[0].start),(u[1].stop,u[0].stop),(0,0,0),2)
        for j,v in enumerate(bysize):
            if x_overlaps(u,v):
                if above(u,v):
                    order[i,j] = 1
            else:
                if [w for w in bysize if separates(w,u,v)]==[]:
                    if left_of(u,v): order[i,j] = 1
    return order

def processOneLineImage(gray_img, iTag):
    (_, img) = cv2.threshold(gray_img, 110, 255, cv2.THRESH_BINARY_INV)
    img = img[:, 2:img.shape[1]-2]
    scale = psegutils.estimate_scale(img)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
    closed = cv2.dilate(img, kernel, iterations = 1)
    edges = cv2.Canny(closed,60,300)
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(edges,contours,-1,(255,255,255),1)
    #cv2.imwrite('edges%s.png' % iTag,edges)
    boxmap = psegutils.compute_boxmap(img,scale,threshold=(.4,10),dtype='B')
    #combineBoxmap(boxmap)
    cv2.imwrite('box%s.png' % iTag, boxmap*255)
    h_projection = hprojection(boxmap*255)
    top, bottom = cropProjection(h_projection)
    regions = splitProjection(h_projection, top, bottom,30,2)
    #print iTag, top,bottom
    #print regions
    #print v_projection[1270:1450]
    if len(iTag) == 0:
        return regions,top,bottom
    for region in regions:
        topStart, TopEnd = region
        cr_img =cv2.getRectSubPix(gray_img, (gray_img.shape[1]-4, TopEnd-topStart+8), (gray_img.shape[1]/2, (TopEnd+topStart)/2))
        cv2.imwrite('%sx%d.png' % (iTag, topStart), cr_img)
    return regions,top,bottom


def processOnePageImage(gray_img, iTag, rotationAngel=0):
    (_, img) = cv2.threshold(gray_img, 110, 255, cv2.THRESH_BINARY_INV)
    #cv2.imwrite('crop1%s.png' % iTag, img)
    scale = psegutils.estimate_scale(img)
    binary = remove_hlines(img, gray_img, scale)
    binary = remove_vlines(binary, gray_img, scale)
    #cv2.imwrite('crop2%s.png' % iTag, binary*255)
    #dftSkew(gray_img)
    if rotationAngel == 0:
        img_crop = interpolation.rotate(gray_img,90)
        angle = estimate_angle(img_crop)
        print iTag,angle
    #    binary = interpolation.rotate(binary,90+angle)
    #    boxmap = psegutils.compute_boxmap(binary,scale,dtype='B')
    #    cv2.imwrite('box%s.png' % iTag, boxmap*255)
        img_crop = interpolation.rotate(img_crop,angle-90, cval=140)
        cv2.imwrite('crop%s.png' % iTag, img_crop)
    elif rotationAngel == -10:
        print "start"
        maxSplit = 0
        minIndex = 20
        for i in range(6):
            img_crop = interpolation.rotate(gray_img,i*0.1,cval=140)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
            closed = cv2.dilate(img_crop, kernel, iterations = 1)
            edges = processImage(closed, 50, 150, tag=iTag)
            regions,left, right,top,bottom = splitImage(edges, img_crop)
            print len(regions),i
            if len(regions) > maxSplit:
                maxSplit = len(regions)
                minIndex = i

        for i in range(5):
            img_crop = interpolation.rotate(gray_img,(i+1)*-0.1,cval=140)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
            closed = cv2.dilate(img_crop, kernel, iterations = 1)
            edges = processImage(closed, 50, 150, tag=iTag)
            regions,left, right,top,bottom = splitImage(edges, img_crop)
            print len(regions), (i+1)*-1
            if len(regions) > maxSplit:
                maxSplit = len(regions)
                minIndex = -1*(i+1)
        print "angle is: %d" % minIndex
        if minIndex == 0:
            return
        img_crop = interpolation.rotate(gray_img,minIndex*0.1,cval=140)
#cv2.imwrite('crop%s.png' % iTag, img_crop)
    else:
        if rotationAngel != 0.01:
            img_crop = interpolation.rotate(gray_img,rotationAngel,cval=140)
            cv2.imwrite('crop%s.png' % iTag, img_crop)
        else:
            img_crop = gray_img
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
    closed = cv2.dilate(img_crop, kernel, iterations = 1)
    edges = processImage(closed, 50, 150, tag=iTag)
    splitImage(edges, img_crop, iTag)

if len(sys.argv) < 3:
    print 'You should tell me the image you want to process!'
    exit(1) 
img = cv2.imread(sys.argv[2])
#kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
#img = cv2.dilate(img, kernel, iterations = 3)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imwrite('gray.png', gray)
#erode(img)
#(_, img) = cv2.threshold(gray, 110, 255, cv2.THRESH_BINARY_INV)
##img = rotate(img, 90)
##img=np.int32(img)
#cv2.imwrite('original.png', img)
##mask = im > im.mean()

#scale = psegutils.estimate_scale(img)
#print scale
#binary = remove_hlines(img,scale)
#cv2.imwrite('label.png', binary*255)
#binary = remove_vlines(binary,scale,3)
#cv2.imwrite('label.png', binary*255)

#gray = np.float32(gray)
#dst = cv2.cornerHarris(gray,2,3,0.04)
#dst = cv2.dilate(dst,None)
#img[dst>0.01*dst.max()]=[0,0,255]

#img=gray
#laplacian = cv2.Laplacian(gray,cv2.CV_8U)
#cv2.imwrite('laplacian.png', laplacian)
#minLineLength = 50
#maxLineGap = 50
#height, width = gray.shape
#theta = 0
#lines = cv2.HoughLinesP(laplacian,1,np.pi/180,100,minLineLength,maxLineGap)
#if lines is not None:
#    for x1,y1,x2,y2 in lines[0]:
#        if x2 == x1:
#            cv2.line(img,(x1,0),(x1,height),(255,0,0),2)
#            theta = 0
#        else:
#            tanTheta = float((y2-y1))/(x2-x1)
#            tmp = np.arctan2(y2-y1, x2-x1) * 180/np.pi
#            if tmp != 0.0 and tmp != 90.0:
#                theta=math.fabs(tmp)
#            else:
#                theta = 0
#            print theta, y2-y1, x2-x1,'hough'
#            ystart=int((0-x1)*tanTheta+y1)
#            yend = int((width-x1)*tanTheta + y1)
#            #cv2.line(img,(0,ystart),(width,yend),(0,255,0),2)
#            cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)
#            cv2.putText(img,"%d,%d,%d,%d" % (x1,y1, x2,y2), (x1,y1), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,0,0))
#cv2.imwrite('hough.png', img)

#sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=-1)
#sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=-1)
#cv2.imwrite('sobelx.png', sobelx)
#cv2.imwrite('sobely.png', sobely)
#gradient = cv2.subtract(sobelx, sobely)
#gradient = cv2.convertScaleAbs(gradient)
#cv2.imwrite('gradient.png', gradient)
#plt.subplot(2,2,1),plt.imshow(img,cmap = 'gray')
#plt.title('Original'), plt.xticks([]), plt.yticks([])
#plt.subplot(2,2,2),plt.imshow(laplacian,cmap = 'gray')
#plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
#plt.subplot(2,2,3),plt.imshow(sobelx,cmap = 'gray')
#plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
#plt.subplot(2,2,4),plt.imshow(sobely,cmap = 'gray')
#plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
#plt.show()

#combine two neighbored image to one
#two params: 1. the left image file path 2. the right image file path
#return: the image with the left image file path
if sys.argv[1] == "combine":
    img1 = cv2.imread(sys.argv[2])
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.imread(sys.argv[3])
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    img3=np.hstack((img1, img2[:,2:]))
    cv2.imwrite(sys.argv[2], img3)
    os.remove(sys.argv[3])
    exit(1)

#split the file vertically
#two params: 1. the image to be splitted 2. the position to split
#return: two image file-- the first is the image with the width equal to the position. the second is the image next to the first image
#three params: 1. the image to be splitted 2. the start position to split 3. the end position to split
#return: one image file: the image with the width between the start position and the end position
if sys.argv[1] == "crop":
    img1 = cv2.imread(sys.argv[2])
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    endPos=int(sys.argv[3])
    if len(sys.argv) ==5:
        startPos = int(sys.argv[4])
        cv2.imwrite(sys.argv[2][0:sys.argv[2].rfind('.')]+'-'+str(endPos+2)+'.png', img1[:,endPos:startPos])
        exit(1)
    cv2.imwrite(sys.argv[2], img1[:,0:endPos])
    index = sys.argv[2].rfind('-')
    endIndex = sys.argv[2].rfind('.')
    fileExtension = sys.argv[2][endIndex:]
    pos = int(sys.argv[2][index+1:endIndex])+endPos
    newfile = sys.argv[2][:index+1]+str(pos+2)+fileExtension
    cv2.imwrite(newfile, img1[:,endPos:])
    exit(1)

#split the file horizontally
#two params: 1. the image to be splitted 2. the position to split
#return: two image file-- the first is the image with the width equal to the position. the second is the image next to the first image
#three params: 1. the image to be splitted 2. the start position to split 3. the end position to split
#return: one image file: the image with the width between the start position and the end position
if sys.argv[1] == "cropV":
    img1 = cv2.imread(sys.argv[2])
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    endPos=int(sys.argv[3])
    if len(sys.argv) ==5:
        startPos = int(sys.argv[4])
        cv2.imwrite(sys.argv[2][0:sys.argv[2].rfind('.')]+'x'+str(endPos+4)+'.png', img1[endPos:startPos, 2:])
        exit(1)
#    cv2.imwrite(sys.argv[2], img1[:,0:endPos])
#    index = sys.argv[2].rfind('-')
#    endIndex = sys.argv[2].rfind('.')
#    fileExtension = sys.argv[2][endIndex:]
#    pos = int(sys.argv[2][index+1:endIndex])+endPos
#    newfile = sys.argv[2][:index+1]+str(pos+2)+fileExtension
#    cv2.imwrite(newfile, img1[:,endPos:])
    exit(1)

if sys.argv[1] == "splitLine":
    filename = os.path.basename(sys.argv[2])
    endIndex = filename.rfind('.')
    tag = filename[:endIndex]
    processOneLineImage(gray, tag)
    exit(1)

if sys.argv[1] == "dft":
    dftSkew(gray)
    exit(1)

if sys.argv[1] == "path":
    for parent,dirnames,filenames in os.walk("/Users/baidu/Documents/sourcecode/ocropy"):
        for filename in filenames:
            oldName= os.path.join(parent,filename)
#            if oldName.find("crop") > -1:
#                newName=oldName.replace("right","right-").replace("left","left-")
#                os.rename(oldName, newName)
            if filename == "cropleft-.png" or filename == "cropright-.png":
                print filename
                newName=oldName.replace("right-","right").replace("left-","left")
                os.rename(oldName, newName)

    exit(1)

if len(sys.argv) > 3 and len(sys.argv[3]) > 0:
    rotationAngel = 0
    if len(sys.argv)>4:
        rotationAngel = float(sys.argv[4])
    processOnePageImage(gray, sys.argv[3],rotationAngel)
    exit(1)

def processTwoPageImage():
    blur = cv2.GaussianBlur(gray,(3,3),0)
    cv2.imwrite('blur.png', blur)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
    closed = cv2.dilate(blur, kernel, iterations = 1)
    cv2.imwrite('closed.png', closed)

    #(_, img2) = cv2.threshold(blur, 64, 255, cv2.THRESH_BINARY_INV)
    edges = processImage(closed, 60, 250)
    regions,left, right,top,bottom = splitImage(edges, blur)


    height, width = edges.shape
    left = regions[0][0]
    right = regions[-1][1]
    if right <= width/2:
        leftEnd = right
        right = -1
    elif left>=width/2:
        rightStart = left
        left = -1
    else:
        leftEnd, rightStart = findMiddleRegion(regions, width /2)
        if leftEnd == -1 and rightStart == -1:
            leftEnd, rightStart = findMiddleRegion(regions, width /2, 15)

    margin=12
    if left > -1:
        img_crop=cv2.getRectSubPix(gray, (leftEnd-left+margin, bottom-top), ((leftEnd+left)/2, (top+bottom)/2))
        processOnePageImage(img_crop, 'left')

    if right > -1:
        img_crop=cv2.getRectSubPix(gray, (right-rightStart+margin, bottom-top), ((right+rightStart)/2, (top+bottom)/2))
        processOnePageImage(img_crop, 'right')

processTwoPageImage()