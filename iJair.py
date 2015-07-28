#!/usr/bin/python

import numpy as np
import cv2
import math
import sys
from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure import SoftmaxLayer

def removeWhiteBackground(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    blah = img[:,:,1].copy()
    img[img[:,:,1] > 75, 1] = 250
    img[img[:,:,1] > 75, 2] *= 0.8
    img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
    img[blah[:,:] < 75] = [0, 0, 0]
    return img

def getCentroid(img):
   cx, cy = 0, 0
   numPixels = 0
   for i in range(0, img.shape[0]):
       for j in range(0, img.shape[1]):
           if (img[i][j] != [0, 0, 0]).any():
               cx += i
               cy += j
               numPixels += 1
   return cx/numPixels, cy/numPixels

def resistorSize(img, x):
    numPixels = 0
    for j in range(0, img.shape[0]):
        if (img[j][x] != [0, 0, 0]).any():
            numPixels += 1
    return numPixels

def rotate(img, centroid, angle):
    M = cv2.getRotationMatrix2D((centroid[1],centroid[0]),angle,1)
    return cv2.warpAffine(img,M,(img.shape[1], img.shape[0]))

def getBorders(img):
   mx, my = img.shape[0], img.shape[1]
   Mx, My = 0, 0
   for i in range(0, img.shape[0]):
       for j in range(0, img.shape[1]):
           if (img[i][j] != [0, 0, 0]).any():
               if mx > i: mx = i
               if my > j: my = j
               if Mx < i: Mx = i
               if My < j: My = j
   return mx, my, Mx, My

def extractFeature(imgFilePath, div, show = False):
    img = cv2.imread(imgFilePath)
    if show:
        cv2.imshow('ori1', img)

    img = removeWhiteBackground(img)

    centroid = getCentroid(img)
    
    maxSize = 0
    maxAngle = 0
    for angle in range(0, 360):
        angleSize = resistorSize(rotate(img, centroid, angle), centroid[1])
        if angleSize > maxSize:
            maxSize = angleSize
            maxAngle = angle
    img = rotate(img, centroid, maxAngle)

    borders = getBorders(img);
    img = img[borders[0]:borders[2], borders[1]:borders[3]]

    if show:
        cv2.imshow('ori2', img)

    print resistorSize(img, centroid[1])
    
    l = 200
    h = 100
    img = cv2.resize(img, (h, l))

    feature = []
    for k in range(0, div):
        cor = [0, 0, 0]
        s = 1
        for x in range(k*(l/div), (k+1)*(l/div)):
            for y in range(0, h):
                if (img[x][y] != [0, 0, 0]).all():
                    s += 1
                cor += img[x][y]
        corMean = cor/s
        feature.append(corMean[0]);
        feature.append(corMean[1]);
        feature.append(corMean[2]);
        if show:
            img[k*(l/div):(k+1)*(l/div),:] = cor/s
    if show:
        cv2.imshow(imgFilePath, img)
    return feature

extractFeature(sys.argv[1], 50, show = True)

#trainFeatures = []
#folder = sys.argv[1]
#
#for i in range(1, 20):
#    print "Extracting feature from "+str(i)+".jpg"
#    trainFeatures.append(extractFeature(folder+str(i)+".jpg", 50))
#
#print "Reading resistence data"
#resValues = [int(line.rstrip('\n')) for line in open('resValues')]
#
#net = buildNetwork(len(trainFeatures[0]), 75, 1, bias = True)
#ds = SupervisedDataSet(len(trainFeatures[0]), 1)
#
#for i in range(0, len(resValues)):
#    ds.appendLinked(trainFeatures[i], [resValues[i]])
#
#print "Training neural network"
#trainer = BackpropTrainer(net, ds)
#trainer.trainUntilConvergence()
#
#print net.activate(extractFeature("fotos/6.jpg", 50))
#print net.activate(extractFeature("fotos/1.jpg", 50))
cv2.waitKey()
