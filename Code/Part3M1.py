# -*- coding: utf-8 -*-

import os
import numpy as np
from PIL import * #Image
import PIL.Image
import skimage
from skimage.viewer import ImageViewer
import sys
import argparse
from matplotlib import pyplot as plt
from pylab import *
import base64

#import cv2


path = os.getcwd()
os.chdir(path)



#For loop needs to start here

#Image Import
FileName = "6812239_c479fd5691f4bd53cc8de7d651e46f70.jpg"

imagereg = Image.open(FileName)
imgplt = plt.imread(FileName)
imgnp = np.array(Image.open(FileName) )
'''
#Processing

def GrayProcessing(file):
    GS = file.convert('L') #Gray scale
    img_arr = np.array(GS)
    return img_arr
#print(img_arr) #readable format

GSCall = GrayProcessing(imagereg)


#Binary of image
def contourIMG(file):
    plt.figure()
    plt.gray()
    plt.contour(file, origin = 'upper')
    plt.axis('on')
    


def GrayHistogram(file):
    plt.figure()
    plt.hist(file.flatten(), 64)
    plt.axis('on')


def ImageSize(file):
    width, height = file.size
    print('Image Width: ', width, ' pixels')
    print('Image Height: ', height, ' pixels')

def ImageSize2(file):
    width, height = file.size
    return height
    
ImageSize(imagereg)   
GrayHistogram(GSCall) 
contourIMG(GSCall)
    



def ImageSizeHeight(file):
    width, height = file.size
    return height

def npImgArr(file):
    array = np.array(file.getdata() )
    return array

def npMatPlotColour(arr, file):
    figure, xaxis = plt.subplots(figsize = (6,3) )
    xaxis.set_xlim(0, 256) #Image(file)
    data, bincount, edging = xaxis.hist(arr, bins = range(256), edgecolor = 'none')
    #n = matplotlib.colors.Normalize(vmin = bincount.min(), vmax = bincount.max() ) 

npMatPlotColour(npImgArr(imagereg), imagereg)

#figure()
#a = np.array(imagereg.getdata() )
#print(a)
 




###RGB Histogram
def RGBHistogram(file):
    plt.figure()
    color = ('b','g','r')
    for i, RGB in enumerate(color):
        plt.xlim([0, 256])
        plot = cv2.calcHist([file], [i], None, [256], [0, 256])
        plt.plot(plot, color = RGB)
    plt.show()

RGBHistogram(imgplt)
'''
p = '\\'
print(path, p)
p2 = (path, p, imagereg)
print(os.path.getsize(p2) )
#print(os.stat(imagereg).st_size)