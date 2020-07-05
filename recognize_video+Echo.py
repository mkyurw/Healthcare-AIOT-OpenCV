# USAGE
# python recognize_video.py --detector face_detection_model \
#    --embedding-model openface_nn4.small2.v1.t7 \
#    --recognizer output/recognizer.pickle \
#    --le output/le.pickle

# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import pickle
import time
import cv2
import os

# import Echo packages
import RPi.GPIO as GPIO
import sys
import time
import requests

# import SQL packages
import mysql.connector as mariadb
from mysql.connector import Error
import random   #fake SBP,DBP,GLU

pinECHO = 23
pinTRIG = 24
GPIO.setmode(GPIO.BCM)
GPIO.setup(pinECHO, GPIO.IN)
GPIO.setup(pinTRIG, GPIO.OUT)
    
#ESP01S
def DetectorON():
    #url = 'http://192.168.0.16/on'
    url = 'http://192.168.10.196/on'
    print(url)
    headers = {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:76.0) Gecko/20100101 Firefox/76.0'}
    response = requests.get(url,headers=headers)
    print(response)
    print('{}'.format(response.text))
    
def DetectorOFF():
    #url = 'http://192.168.0.16/off'
    url = 'http://192.168.10.196/off'
    print(url)
    headers = {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:76.0) Gecko/20100101 Firefox/76.0'}
    response = requests.get(url,headers=headers)
    print(response)
    print('{}'.format(response.text))

    
####
def pulseIn(pin):
    if GPIO.wait_for_edge(pin, GPIO.RISING, timeout=500) is None:
        return 0
    
    start_time = time.time()
    GPIO.wait_for_edge(pin, GPIO.FALLING, timeout=500)
    return (time.time() -start_time) *1000000

try:
    sonar = True
    while sonar == True:
        GPIO.output(pinTRIG,0)
        time.sleep(2.0/1000000)
        
        GPIO.output(pinTRIG,1)
        time.sleep(10.0/1000000)
        GPIO.output(pinTRIG,0)
        
        d= pulseIn(pinECHO) / 28.9 /2
        
        
        if d > 400 or d == 0:
            print("something wrong")
            continue
        if d < 40 :                              # if someone close more than 40cm, turn on
            print("somebody in fornt of here.")
            print(str(d)+" cm")
            
            ##Relay
            print('Relay is ON...')
            DetectorON()
            # script for detectors
            # construct the argument parser and parse the arguments
            detector = 'face_detection_model'
            embedding_model = 'openface_nn4.small2.v1.t7'
            recognizer = 'output/recognizer.pickle'
            le = 'output/le.pickle'
            custom_confidence = 0.5

            # load our serialized face detector from disk
            print("[INFO] loading face detector...")
            protoPath = os.path.sep.join([detector, "deploy.prototxt"])
            modelPath = os.path.sep.join([detector,
                "res10_300x300_ssd_iter_140000_fp16.caffemodel"])
            detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

            # load our serialized face embedding model from disk
            print("[INFO] loading face recognizer...")
            embedder = cv2.dnn.readNetFromTorch(embedding_model)

            # load the actual face recognition model along with the label encoder
            recognizer = pickle.loads(open(recognizer, "rb").read())
            le = pickle.loads(open(le, "rb").read())

            # initialize the video stream, then allow the camera sensor to warm up
            print("[INFO] starting video stream...")
            vs = VideoStream(src=1).start()
            #vs = VideoStream(usePiCamera=True).start()
            time.sleep(2.0)

            # start the FPS throughput estimator
            fps = FPS().start()

            # loop over frames from the video file stream
            hasDetected = False
            notice = False
            while True:
                #print("start loop")
                # grab the frame from the threaded video stream
                frame = vs.read()

                # resize the frame to have a width of 600 pixels (while
                # maintaining the aspect ratio), and then grab the image
                # dimensions
                frame = imutils.resize(frame, width=600)
                (h, w) = frame.shape[:2]

                # construct a blob from the image
                imageBlob = cv2.dnn.blobFromImage(
                    cv2.resize(frame, (300, 300)), 1.0, (300, 300),
                    (104.0, 177.0, 123.0), swapRB=False, crop=False)

                # apply OpenCV's deep learning-based face detector to localize
                # faces in the input image
                detector.setInput(imageBlob)
                detections = detector.forward()

                # loop over the detections
                for i in range(0, detections.shape[2]):
                    # extract the confidence (i.e., probability) associated with
                    # the prediction
                    confidence = detections[0, 0, i, 2]

                    # filter out weak detections
                    
                    if confidence > custom_confidence:
                        
                        # compute the (x, y)-coordinates of the bounding box for
                        # the face
                        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                        (startX, startY, endX, endY) = box.astype("int")

                        # extract the face ROI
                        face = frame[startY:endY, startX:endX]
                        (fH, fW) = face.shape[:2]

                        # ensure the face width and height are sufficiently large
                        if fW < 20 or fH < 20:
                            continue

                        # construct a blob for the face ROI, then pass the blob
                        # through our face embedding model to obtain the 128-d
                        # quantification of the face
                        faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
                            (96, 96), (0, 0, 0), swapRB=True, crop=False)
                        embedder.setInput(faceBlob)
                        vec = embedder.forward()

                        # perform classification to recognize the face
                        preds = recognizer.predict_proba(vec)[0]
                        j = np.argmax(preds)
                        proba = preds[j]
                        name = le.classes_[j][2:]
                        id = le.classes_[j][:1]
                        # draw the bounding box of the face along with the
                        # associated probability
                        text = "{}: {:.2f}%".format(name, proba * 100)
                        y = startY - 10 if startY - 10 > 10 else startY + 10
                        cv2.rectangle(frame, (startX, startY), (endX, endY),
                            (0, 0, 255), 2)
                        cv2.putText(frame, text, (startX, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
                # update the FPS counter
                fps.update()

                # show the output frame
                cv2.imshow("Frame", frame)
                key = cv2.waitKey(1) & 0xFF
                
                if (not notice):
                    notice = True
                    print(f'Your name is: {name} Patno is: {id}')
                    print('press \'y\' to start detection')
                    print('press \'r\' to restart detection')
                    print('press \'q\' to quit detection')
                if key == ord("y"):
                    if (not hasDetected):
                        hasDetected = True
                        # Insert to DB
                        connection = mariadb.connect(host='database-1.cr4dw42pglz1.ap-northeast-1.rds.amazonaws.com', user='admin', password='iiiAIOT06', database='healthcare')
                        cursor = connection.cursor()
                        if connection.is_connected():
                            print("連線成功")
                            print(f'Your name is: {name} Patno is: {id}')
                            value1 = int(id)
                            value2 = name
                            sbp = str(random.randint(110,170))  
                            dbp = str(random.randint(60,120))
                            glu = str(random.randint(80,130))
                            query = "INSERT INTO dailymeasure VALUES (%s,%s,now(),%s,%s,%s,0,0);"
                            #cursor.execute(query,(value1,value2,sbp,dbp,glu))
                            connection.commit()
                            print("Insert成功")
                            print('detect start')
                            # print(id)
                            url = 'http://192.168.10.125/sw?patno={}'.format(id)
                            print(url)
                            headers = {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:76.0) Gecko/20100101 Firefox/76.0'}
                            response = requests.get(url,headers=headers)
                            print(response)
                            print('response html is: \n {}'.format(response.text))
                            print('detect end')
                            print('press \'r\' to restart detection')
                            print('press \'q\' to quit detection')
                            
                        else:
                            print("連線失敗")
                
                # if the `q` key was pressed, break from the loop
                if key == ord("r"):
                    notice = False
                    hasDetected = False
                    print("restart")
                    continue
                if key == ord("q"):
                    sonar = False
                    DetectorOFF()
                    vs.stop()
                    break
            
            # stop the timer and display FPS information
            fps.stop()
            print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
            print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

        else:
            
            print ("Distance: " +str(d)+" cm")
            time.sleep(0.5)
except:
    print('something is wrong')
    pass

print('Bye~')
cv2.destroyAllWindows()
GPIO.cleanup()
DetectorOFF()
vs.stop()

