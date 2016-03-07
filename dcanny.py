#!/usr/bin/python
# -*- coding: utf-8 -*-
import cv2
import cv
import numpy as np
import os,shutil,sys
from scipy.ndimage import measurements,interpolation,morphology,filters
import re
import codecs
import fontforge

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


# initialize the list of reference points and boolean indicating
# whether cropping is being performed or not
refPt = []
cropping = False
img=None
clone=None
charIndex = 0
files=[]
def click_and_crop(event, x, y, flags, param):
    # grab references to the global variables
    global refPt, cropping,img,clone
    
    # if the left mouse button was clicked, record the starting
    # (x, y) coordinates and indicate that cropping is being
    # performed
    if event == cv2.EVENT_LBUTTONDOWN:
        refPt = [(x, y)]
        cropping = True
    elif event == cv2.EVENT_MOUSEMOVE and cropping == True:
        simg=img.copy()
        cv2.rectangle(simg, refPt[0], (x,y), (0, 255, 0), 1)
        cv2.imshow("canny demo", simg)
    # check to see if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
        # record the ending (x, y) coordinates and indicate that
        # the cropping operation is finished
        refPt.append((x, y))
        cropping = False
        
        # draw a rectangle around the region of interest
        print refPt
        simg = img.copy()
        cv2.rectangle(simg, refPt[0], refPt[1], (0, 255, 0), 1)
        cv2.imshow("canny demo", simg)

def loadImage(filePath):
    global img, refPt,clone
    refPt = []
    img = cv2.imread(filePath)
    clone=img.copy()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    if height > 700:
        img = interpolation.rotate(img,90)
    cv2.imshow('canny demo',img)

def showImage(filePath, fileIndex, char=''):
    global img, refPt,clone,charIndex,files
    loadImage(filePath)
    height, width,_ = clone.shape
    zclone=clone.copy()
    print height,width

    pos=filePath.find('x')
    posEnd=filePath.rfind('.')
    imgStart = int(filePath[pos+1:posEnd])
    while True:
        k=cv2.waitKey(0)
        if k == 27:
            cv2.destroyAllWindows()
            exit(1)
        elif k==ord('d'):
            #delete the current file
            #os.remove(filePath)
            charIndex-=1
            shutil.move(filePath, filePath+'deleted.png')
            print 'delete %s' % filePath
            break
        elif k==ord('f'):
            img = cv2.imread(filePath[0:pos]+'.png')
            clone=img.copy()
            height,width,_=clone.shape
            cv2.imshow('canny demo',img)
            files.insert(fileIndex+1, filePath)
            imgStart = imgStart - 10
            if imgStart < 0:
                imgStart = 0
            filePath = filePath[0:pos+1]+str(imgStart)+'.png'
        elif k==ord('e'):
            img = cv2.imread(filePath[0:pos]+'.png')
            newStart=imgStart-200
            if newStart <0:
                newStart = 0
            newEnd = imgStart+height+200
            if newEnd > img.shape[0]:
                newEnd = img.shape[0]
            img= img[newStart:newEnd].copy()
            clone=img.copy()
            height,width,_=clone.shape
            cv2.imshow('canny demo',img)
        elif k == ord('S'):
            #save the image
            if len(refPt) == 2:
                shutil.move(filePath, filePath+'deleted.png')
                newFile = filePath[0:pos+1]+str(imgStart+refPt[0][1])
                cv2.imwrite(newFile+'.png', clone[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]])
                shutil.move(newFile+'.png', newFile+'y'+char+'.png')
        elif k == ord('s'):
            cv2.imwrite(filePath, clone)
            shutil.move(filePath, filePath[0:-4]+'y'+char+'.png')
            break
        elif k == ord('z'):
            #reset the mouse clip
            img = zclone.copy()
            clone=img.copy()
            height, width,_ = clone.shape
            refPt = []
            cv2.imshow('canny demo',img)
        elif k == ord('l'):
            img=clone[:,1:].copy()
            clone=img.copy()
            width-=1
            cv2.imshow('canny demo',img)
        elif k == ord('r'):
            img=clone[:,0:-1].copy()
            clone=img.copy()
            width-=1
            cv2.imshow('canny demo',img)
        elif k == ord('T'):
            img=clone[5:,:].copy()
            clone=img.copy()
            height-=5
            cv2.imshow('canny demo',img)
        elif k == ord('t'):
            img=clone[1:,:].copy()
            clone=img.copy()
            height-=1
            cv2.imshow('canny demo',img)
        elif k == ord('B'):
            img=clone[0:-5,:].copy()
            clone=img.copy()
            height-=5
            cv2.imshow('canny demo',img)
        elif k == ord('b'):
            img=clone[0:-1,:].copy()
            clone=img.copy()
            height-=1
            cv2.imshow('canny demo',img)
        elif k == ord('c'):
            #cut the image
            y = refPt[0][1]
            if height>700:
                y = refPt[0][0]
            newFile = filePath[0:pos+1]+str(imgStart+y)
            cv2.imwrite(filePath, clone[0:y, :])
            cv2.imwrite(newFile+'.png', clone[y:, :])
            files.insert(fileIndex+1, newFile+'.png')
            loadImage(filePath)
            height, width,_ = clone.shape
            zclone=clone.copy()
        elif k == 63233:
            break

cv2.namedWindow('canny demo', cv.CV_WINDOW_AUTOSIZE)
cv2.setMouseCallback("canny demo", click_and_crop)
#for fpathe,dirs,fs in os.walk('/Users/baidu/Documents/sourcecode/ocropy'):
#    for f in fs:
#        filePath=os.path.join(fpathe,f)
#        if filePath.find("jusongguangyun2") >0 and filePath.find("crop") >0 and  filePath.find("x") >0 and filePath.find("delete") <0 and filePath.endswith('png'):
#            print filePath
#            showImage(filePath)
def getKey(filename):
    arr = re.split('-|x', filename[:-4])[2:]
    ret = []
    for i in arr:
        ret.append(int(i))
    return ret


def cmpFile(filename1, filename2):
    keys1 = getKey(filename1)
    keys2 = getKey(filename2)
    for i in range(min(len(keys1),len(keys2)) -1):
        if keys1[i] != keys2[i]:
            return cmp(keys2[i],keys1[i])
    return cmp(keys1[-1],keys2[-1])

def cmpFileIncludingChar(filename1, filename2):
    keys1 = getKey(filename1[:filename1.rfind('y')]+'.png')
    keys2 = getKey(filename2[:filename2.rfind('y')]+'.png')
    for i in range(min(len(keys1),len(keys2)) -1):
        if keys1[i] != keys2[i]:
            return cmp(keys2[i],keys1[i])
    return cmp(keys1[-1],keys2[-1])

def listFile(dirname, includingChar=False):
    rightfiles = []
    leftfiles = []
    charCount = 0
    for filename in os.listdir(dirname):
        filePath = os.path.join(dirname,filename)
        if os.path.isfile(filePath) and filename.find('x') > 0 and filename.find('png') > 0  and filename.find('deleted')<0:
            if includingChar:
                if filename.find('y') > 0:
                    if filename.find('right') > 0:
                        rightfiles.append(filePath)
                    elif filename.find('left') > 0:
                        leftfiles.append(filePath)
            else:
                if filename.find('y') > 0:
                    charCount+=1
                elif filename.find('right') > 0:
                    rightfiles.append(filePath)
                elif filename.find('left') > 0:
                    leftfiles.append(filePath)
    
    if includingChar:
        rightfiles = sorted(rightfiles, cmp=cmpFileIncludingChar)
        leftfiles = sorted(leftfiles, cmp=cmpFileIncludingChar)
    else:
        rightfiles = sorted(rightfiles, cmp=cmpFile)
        leftfiles = sorted(leftfiles, cmp=cmpFile)
    rightfiles.extend(leftfiles)
    return rightfiles,charCount

def get_wide_ordinal(char):
    print len(char)
    if len(char) != 2:
        return ord(char)
    return 0x10000 + (ord(char[0]) - 0xD800) * 0x400 + (ord(char[1]) - 0xDC00)

fh = codecs.open('/Users/baidu/Documents/sourcecode/ocropy/guangyun-text/guangyun.txt','r','utf-8')
lines=fh.readlines()
fh.close()

fileIndex = 0
#lastIndex=0
#guangyunGuji = codecs.open('guangyun.guji','w','utf-8')
#for i in range(len(lines)/5):
#    pos = lines[i*5+1].find("\">")
#    posEnd = lines[i*5+1].find("</")
#    char = lines[i*5+1][pos+2:posEnd]
#    pos = lines[i*5+3].find("\">")
#    posEnd = lines[i*5+3].find("</")
#    lastCharIndex=lines[i*5+3][pos+2:posEnd]
#    codepoint=get_wide_ordinal(lastCharIndex)
#    if codepoint != lastIndex:
#        guangyunGuji.write(u'○')
#        lastIndex = codepoint
#    guangyunGuji.write(char)
#    guangyunGuji.write('\anno{')
#    pos = lines[i*5+4].find(">")
#    posEnd = lines[i*5+4].find("</")
#    lastCharIndex=lines[i*5+4][pos+1:posEnd]
#    guangyunGuji.write(lastCharIndex)
#    guangyunGuji.write('}')
#guangyunGuji.close()

if len(sys.argv)>1 and sys.argv[1] == "makeFont":
    for book in range(1, 6):
        for page in range(1, 50):
            dirpath = '/Users/baidu/Documents/sourcecode/ocropy/jusongguangyun%d-%d' % (book,page)
            if os.path.isdir(dirpath):
                if book == 1 and page < 5:
                    pass
                else:
                    right, charCount= listFile(dirpath, True)
                    files.extend(right)
                    charIndex+=charCount
    os.system('cp blank.sfd guangyun.sfd')
    font = fontforge.open('guangyun.sfd')
    font.familyname='guangyun'.decode('utf-8')
    font.fondname='guangyun'.decode('utf-8')
    font.fontname='guangyun'.decode('utf-8')
    font.fullname='鉅宋重修廣韵体'.decode('utf-8')
    for file in files:
        char=file[file.rfind('y')+1:file.find('.')]
        codepoint=get_wide_ordinal(char.decode('utf-8'))
        print char,codepoint
        glyph=font.createChar(codepoint,"uni"+char)
        layer=glyph.foreground;
        if layer.isEmpty():
            pnm = file.replace('png', 'pnm')
            svg = file.replace('png', 'svg')
            os.system('convert %s %s' % (file, pnm))
            os.system('potrace -s %s' % pnm)
            shutil.move(svg, svg[:svg.rfind('y')]+'.svg')
            svg = svg[:svg.rfind('y')]+'.svg'
            glyph.importOutlines(svg)
    font.generate('guangyun.otf')
    font.save()
else:
    for book in range(1, 6):
        for page in range(1, 50):
            dirpath = '/Users/baidu/Documents/sourcecode/ocropy/jusongguangyun%d-%d' % (book,page)
            if os.path.isdir(dirpath):
                if book == 1 and page < 6:
                    pass
                else:
                    right, charCount= listFile(dirpath)
                    files.extend(right)
                    charIndex+=charCount
    for file in files:
        charIndex+=1
        pos = lines[charIndex*5+1].find("\">")
        posEnd = lines[charIndex*5+1].find("</")
        char = lines[charIndex*5+1][pos+2:posEnd]
        print char,charIndex,file
        showImage(file,fileIndex,char)
        fileIndex+=1



