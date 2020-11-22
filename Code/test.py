from sys import platform
import numpy as np
import cv2
import pickle
import tensorflow as tf
from matplotlib import pyplot as plt
nameImage="Image_test/anh12.jpg"
img=cv2.imread(nameImage, 1)
img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(img, 190, 200)
(h, w) = img.shape
for i in range(h):
    for j in range(w):
        G = edges[i, j]
        if G==0:
            img[i, j]=255
cv2.imshow("anh", img)

cv2.waitKey(0)