# from django.db import models
from distutils.command.upload import upload
from email.policy import default
from io import BytesIO
from json import load
from sqlite3 import Timestamp
from django.db import models
from django.forms import IntegerField
from django.core.files.base import ContentFile
from keras.models import load_model

from PIL import Image
import cv2 
import numpy as np
import face_recognition
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic


def gray(img):
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    return img

def faceDetection(img):
    faceCascade=cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_alt.xml")
    imgGray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=faceCascade.detectMultiScale(imgGray,1.1,4)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

    return img

def MoodDetection(imginp):
    img=imginp
    model=load_model("image\emotionnew.h5")
    output=""
    labels=["Happy","angry","sad","fear"]
    faceCascade=cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_alt.xml")
    imgGray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # img=cv2.resize(img,[150,150])

    # img1=np.array([img])
    # pred=np.argmax(model.predict(img1))


    faces=faceCascade.detectMultiScale(imgGray,1.1,4)
    
    for (x,y,w,h) in faces:
        cv2.rectangle(imginp,(x,y),(x+w,y+h),(255,0,0),2)
        face_img=img[x:x+w,y:y+h]
        face_img=cv2.resize(img,[150,150])
        face_img=np.array([face_img])
        # pred=np.argmax(model.predict(img1))
        pred=np.argmax(model.predict(face_img))
        cv2.putText(img, labels[np.argmax(pred)] , (x, y+20), cv2.FONT_HERSHEY_PLAIN, 2, (255,255,255), 2)

        output+=" "+labels[np.argmax(pred)]
    return imginp,output

def Gender(imginp):
    img1=imginp
    model=load_model("image\gender240.h5")
    output=""
    labels=["Female","Male"]

    img1=cv2.resize(img1,[240,240])
    img1=np.array([img1])
    pred=np.argmax(model.predict(img1))
    output=labels[np.argmax(pred)]

    return imginp,output
    
    
def Mask(img):
    net = cv2.dnn.readNet('image/yolov4-custom_best.weights', 'image/yolov4-custom.cfg')
    classes=["Wearing mask","Not wearing mask"]

    # img=imginp 
    font = cv2.FONT_HERSHEY_PLAIN
    colors = np.random.uniform(0, 255, size=(100, 3))
    img=cv2.resize(img,[800,600],interpolation=cv2.INTER_AREA)
    height, width, _ = img.shape

    blob = cv2.dnn.blobFromImage(img, 1/255, (416, 416), (0,0,0), swapRB=True, crop=False)
    net.setInput(blob)
    output_layers_names = net.getUnconnectedOutLayersNames()
    layerOutputs = net.forward(output_layers_names)

    boxes = []
    confidences = []
    class_ids = []

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.2:
                center_x = int(detection[0]*width)
                center_y = int(detection[1]*height)
                w = int(detection[2]*width)
                h = int(detection[3]*height)

                x = int(center_x - w/2)
                y = int(center_y - h/2)

                boxes.append([x, y, w, h])
                confidences.append((float(confidence)))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.4)
    if len(indexes)>0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])


            confidence = str(round(confidences[i],2))
            color = colors[i]
            cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
            cv2.putText(img, label + " " + confidence, (x, y+20), font, 1, (255,255,255), 2)
    
    return img, ""


def face_match(img1,img2):
    image1=img1
    imgTest=img2
    # image1 = face_recognition.load_image_file(img1)
    image1=cv2.cvtColor(image1,cv2.COLOR_BGR2RGB)
    # imgTest=face_recognition.load_image_file(img2)
    imgTest=cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB)

    faceLoc=face_recognition.face_locations(image1)[0]
    encodeim1=face_recognition.face_encodings(image1)[0]
    # cv2.rectangle(image1,(faceLoc[3],faceLoc[4]),(faceLoc[1]),faceLoc[2],(0,0,255),2)

    faceLoc2=face_recognition.face_locations(imgTest)[0]
    encodeim2=face_recognition.face_encodings(imgTest)[0]
    # cv2.rectangle(imgTest,(faceLoc2[3],faceLoc2[4]),(faceLoc2[1]),faceLoc2[2],(0,0,255),2)

    results=face_recognition.compare_faces([encodeim1],encodeim2)
    resultDis=face_recognition.face_distance([encodeim1],encodeim2)
    return img1,str(results)+" "+str(resultDis)

def pose(image):
    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as holistic:

       
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        
        image.flags.writeable = False
        results = holistic.process(image)

        
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        
        image=cv2.flip(image,1)
        return image

# Create your models here.
class images(models.Model):
    img_id=models.AutoField(primary_key=True)
    image=models.FileField(upload_to="images/imagedata",default="")
    image2=models.FileField(upload_to="images/imagedata",default="")
    img_name=models.CharField(max_length=50)
    prediction=models.CharField(max_length=50,default="")
    choice=models.IntegerField(default=0)

    def __str__(self):
        return self.img_name
      
    def save(self,*args,**kwargs):
        open_img=Image.open(self.image)

        cv2_img=np.array(open_img)

        predict=""
        img=cv2_img
        # img=gray(cv2_img)
        if self.choice==0:

            img=faceDetection(cv2_img)
        elif self.choice==1:
            open_img2=Image.open(self.image2)
            cv2_img2=np.array(open_img2)
            # img,predict=face_match(open_img,open_img2)
            img,predict=face_match(cv2_img,cv2_img2)

        elif self.choice==2:
            img,predict=MoodDetection(cv2_img)
        elif self.choice==3:
            img,predict=Gender(cv2_img)
        elif self.choice==4:
            img,predict=Mask(cv2_img)
        elif self.choice==5:
            img=pose(img)

        close_img=Image.fromarray(img)

        buffer=BytesIO()
        close_img.save(buffer,format='png')
        image_png=buffer.getvalue()

        self.prediction=predict
        self.image.save(str(self.image),ContentFile(image_png),save=False)

        super().save(*args,**kwargs)