import numpy as np
import cv2
import pickle
import tensorflow as tf
#############################################
 
frameWidth= 200         # CAMERA RESOLUTION
frameHeight = 200
brightness = 180
threshold = 0.75         # PROBABLITY THRESHOLD
font = cv2.FONT_HERSHEY_SIMPLEX
##############################################

# SETUP THE VIDEO CAMERA
#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10, brightness)
# IMPORT THE TRANNIED MODEL
# pickle_in=open("saved_model.hp5","rb")  ## rb = READ BYTE
# model=pickle.load(pickle_in)
loaded_model = tf.keras.models.load_model("saved_model.hp5")
model =loaded_model
def grayscale(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    return img
def equalize(img):
    img =cv2.equalizeHist(img)
    return img
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
    #else : return 'Khong co logo'
while True:
 
# READ IMAGE

    success, imgOrignal = cap.read()
# PROCESS IMAGE
    img = np.asarray(imgOrignal)
    img = cv2.resize(img, (50, 50))
    img = preprocessing(img)
    cv2.imshow("Processed Image", img)
    img = img.reshape(1, 50, 50, 1)
    cv2.putText(imgOrignal, "Hang xe:" , (20, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(imgOrignal, "Do chinh xac: ", (20, 75), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
# PREDICT IMAGE
    predictions = model.predict(img)
    classIndex = model.predict_classes(img)
    probabilityValue =np.amax(predictions)
    if probabilityValue > threshold:
        print("Hãng xe: " +getCalssName(classIndex))
        cv2.putText(imgOrignal,"    "+str(getCalssName(classIndex)), (120, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(imgOrignal,"    "+ str(round(probabilityValue*100,2) )+"%", (180, 75), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
    else:
        print("Hãng xe: không có ")
        cv2.putText(imgOrignal,"   khong  ", (120, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(imgOrignal,"  100%  ", (180, 75), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.imshow("Result", imgOrignal)
    if cv2.waitKey(1) and 0xFF == ord('q'):
        break