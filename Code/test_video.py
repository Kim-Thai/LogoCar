import numpy as np
import cv2
import pickle
import tensorflow as tf
#############################################
 
frameWidth= 400         # CAMERA RESOLUTION
frameHeight = 400
brightness = 180
threshold = 0.75         # PROBABLITY THRESHOLD
font = cv2.FONT_HERSHEY_SIMPLEX
##############################################

# SETUP THE VIDEO CAMERA
nameVideo="Image_test/toplogo.mp4"
cap = cv2.VideoCapture(nameVideo)
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
    #img = cv2.Canny(img, 50, 200)
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
while True:
 
# READ IMAGE
    success, imgOrignal = cap.read()
    img_resized = cv2.resize(src=imgOrignal, dsize=(frameWidth, frameHeight))

# PROCESS IMAGE
    img = np.asarray(img_resized)
    img = cv2.resize(imgOrignal, (50, 50))
    img = preprocessing(img)
    cv2.imshow("Processed Image", img)
    img = img.reshape(1, 50, 50, 1)
    cv2.putText(img_resized, "Hang xe:" , (10, 35), font, 0.5, (0, 0, 255), 2, cv2.FONT_HERSHEY_COMPLEX)
    cv2.putText(img_resized, "Do chinh xac: ", (10, 75), font, 0.5, (0, 0, 255), 2, cv2.FONT_HERSHEY_COMPLEX)
# PREDICT IMAGE
    predictions = model.predict(img)
    classIndex = model.predict_classes(img)
    probabilityValue =np.max(predictions)
    if probabilityValue > threshold:
        print("Hãng xe: " +getCalssName(classIndex))
        print("Độ chính xác: " +str(round(probabilityValue*100,2)))
        cv2.putText(img_resized," "+str(getCalssName(classIndex)), (90, 35), font, 0.5, (0, 0, 255), 2, cv2.FONT_HERSHEY_COMPLEX)
        cv2.putText(img_resized," "+ str(round(probabilityValue*100,2) )+"%", (130, 75), font, 0.5, (0, 0, 255), 2, cv2.FONT_HERSHEY_COMPLEX)
    else:
        print("Hãng xe: không có ")
        cv2.putText(img_resized,"khong  ", (90, 35), font, 0.5, (0, 0, 255), 2, cv2.FONT_HERSHEY_COMPLEX)
        cv2.putText(img_resized,"100%  ", (130, 75), font, 0.5, (0, 0, 255), 2, cv2.FONT_HERSHEY_COMPLEX)
    cv2.imshow(nameVideo, img_resized)
    if cv2.waitKey(1) and 0xFF == ord('q'):
        break