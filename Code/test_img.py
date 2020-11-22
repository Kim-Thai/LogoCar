from sys import platform
import numpy as np
import cv2
import pickle
import tensorflow as tf
from matplotlib import pyplot as plt
#############################################
 
frameWidth= 400         # CAMERA RESOLUTION
frameHeight = 400
brightness = 180
threshold = 0.75         # PROBABLITY THRESHOLD
font = cv2.FONT_HERSHEY_SIMPLEX
##############################################



loaded_model = tf.keras.models.load_model("saved_model.hp5")
model =loaded_model

# CHUYỂN ẢNH XÁM
def grayscale(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    return img
# tìm biên dùng canny
def equalize(img):
    img =cv2.equalizeHist(img)
    #edges = cv2.Canny(img, 50, 200)
    return img
# GỌI LẠI HÀM

def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    img = img/255
    return img
def getCalssName(classNo):
    if   classNo == 0: return 'Audi'
    elif classNo == 1: return 'BMW'
    elif classNo == 2: return 'Ferrari'
    elif classNo == 3: return 'ford'
    elif classNo == 4: return 'Honda'
    elif classNo == 5: return 'lexus'
    elif classNo == 6: return 'Mazda'
    elif classNo == 7: return 'mercedes'
    elif classNo == 8: return 'mitsubishi'
    elif classNo == 9: return 'alfa romeo'
# READ IMAGE
nameImage="Image_test/anh12.jpg"
################3

##########
imgOrignal =cv2.imread(nameImage, 1)
#img_resized = cv2.resize(src=imgOrignal, dsize=(frameWidth, frameHeight))
# PROCESS IMAGE
# cắt
#img
img = np.asarray(imgOrignal)
img = cv2.resize(img, (50, 50))
img = preprocessing(img)
#cv2.imshow("Processed Image", img)
img = img.reshape(1, 50, 50, 1)
# PREDICT IMAGE
predictions = model.predict(img)
classIndex = model.predict_classes(img)
probabilityValue =np.amax(predictions)
if probabilityValue > threshold:
    print("Hãng xe: " +getCalssName(classIndex))
    print("Độ chính xác : " +str(round(probabilityValue*100,2) ))
else :
    print("Hãng xe: Không có trong data " )
    print("Độ chính xác : 100%" )
cv2.imshow(nameImage, imgOrignal)

cv2.waitKey(0)