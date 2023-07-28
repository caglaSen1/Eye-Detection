import cv2
import numpy as np
import mediapipe as mp
import time

# haar cascade is a pre-trained classifier

cap = cv2.VideoCapture(0)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
writer = cv2.VideoWriter("video_kaydı2.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 20, (width, height))

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")  # cv2.data.haarcascades - the path to where these are stored on our system
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml")


while True:
    success, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Returns the location of all the faces in terms of position:
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)  # detectMultiScale(base img, scaleFactor, minNeighbours, flags, minSize, maxSize)

    # scaleFactor: haar cascade doesn't know how large the image is going to be. So, it 's mean that we may have to change the size of this image,
    # such that the haar cascade has sth that it compare it against you. We need to shrink this base img down and what the scale factor is telling us
    # is how much we should shrink this by at each iteration. Recommended is 1.05 (shrink the img down by 5%)
    # smaller value, higher accuracy but slower performing algorithm
    # larger value, less accuracy but faster performing algorithm

    # minNeighbours: Parameter specifying how many neighbours each candidate rectangle should have to retain it.
    # Haar cascade is return to us a bunch of positions of potential faces. minNeighbours says that how many candidate ractangles do I need
    # that are overlapping in a specific area before I determined that this actually is a face. (This is pretty much saying how accurate does the algo need to be)
    # Higher value results in less detections but with higher quality.
    # 3-6 is good value for it

    # minSize: Obj smaller than that are ignored
    # maxSize: Obj bigger than that are ignored

    for(x, y, w, h) in faces:  # faces gives us a rect
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 5)  # draw a rectangle on the frame
        # roi - region of interest
        roi_gray = gray[y:y+w, x:x+w]  # this'll tell us the location of our face - getting rect (from faces) from our gray scale img and storing in a roi grid.
        roi_color = frame[y:y + h, x:x + w]

        # we're gonna pass roi_gray to our new eye classifier and just draw on roi_color the rect.
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.3, 5)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 5)
        # cv2.rectangle(frame, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 5) - eyes will be on the wrong spot because (ex, ey, ew, eh) is telling us the
        # location of the eyes on our roi_gray img

    cv2.imshow("Frame", frame)
    writer.write(frame)
    if cv2.waitKey(1) & 0xFF == ord("q"): break

cap.release()
writer.release()
cv2.destroyAllWindows()


'''
import cv2
import os
from keras.models import load_model
import numpy as np
from pygame import mixer
import time


mixer.init()
sound = mixer.Sound('alarm.wav')

# set the cascade classifiers:
face = cv2.CascadeClassifier('haar cascade files\haarcascade_frontalface_alt.xml')
leye = cv2.CascadeClassifier('haar cascade files\haarcascade_lefteye_2splits.xml')
reye = cv2.CascadeClassifier('haar cascade files\haarcascade_righteye_2splits.xml')

lbl = ['Close', 'Open']

model = load_model('models/cnncat2.h5')  # loaded the model
path = os.getcwd()
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
count = 0
score = 0  # how long the person has closed his eyes
thicc = 2
rpred = [99]
lpred = [99]

while True:
    success, frame = cap.read()
    height, width = frame.shape[:2]

    # OpenCV algorithm for object detection takes gray images in the input
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # perform the detection:
    # They return an array of detections with x,y coordinates, and height, the width of the boundary box of the object:
    faces = face.detectMultiScale(gray, minNeighbors=5, scaleFactor=1.1, minSize=(25, 25))
    left_eye = leye.detectMultiScale(gray)  # extract only the eyes data from the full image
    right_eye = reye.detectMultiScale(gray)

    cv2.rectangle(frame, (0, height-50), (200, height), (0, 0, 0), thickness=cv2.FILLED)

    for (x, y, w, h) in faces:  # iterate over the faces and draw boundary boxes for each face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (100, 100, 100), 1)

    for (x, y, w, h) in right_eye:  # extracting the boundary box of the eye
        # r_eye only contains the image data of the eye:
        r_eye = frame[y:y+h, x:x+w]  # pull out the eye image from the frame with this code. This will be fed into our CNN classifier which will predict if eyes are open or closed.
        count = count+1

        # the model needs the correct dimensions to start with:
        r_eye = cv2.cvtColor(r_eye, cv2.COLOR_BGR2GRAY)
        r_eye = cv2.resize(r_eye, (24, 24))  # model was trained on 24*24 pixel images
        r_eye = r_eye/255  # normalize our data for better convergence (All values will be between 0-1)
        r_eye = r_eye.reshape(24, 24, -1)
        r_eye = np.expand_dims(r_eye, axis=0)  # Expand the dimensions to feed into our classifier
        rpred = model.predict(r_eye)  # predict each eye with the model
        rpred = np.argmax(rpred, axis=1)
        if rpred[0] == 1:  # eyes are open
            lbl='Open'
        if rpred[0] == 0:  # eyes are closed
            lbl='Closed'
        break

    for (x, y, w, h) in left_eye:
        l_eye = frame[y:y+h, x:x+w]
        count = count+1
        l_eye = cv2.cvtColor(l_eye, cv2.COLOR_BGR2GRAY)
        l_eye = cv2.resize(l_eye, (24, 24))
        l_eye = l_eye/255
        l_eye = l_eye.reshape(24, 24, -1)
        l_eye = np.expand_dims(l_eye, axis=0)
        lpred = model.predict(l_eye)
        lpred = np.argmax(lpred, axis=1)
        if lpred[0] == 1:
            lbl = 'Open'
        if lpred[0] == 0:
            lbl = 'Closed'
        break

    # Calculate Score to Check whether Person is Drowsy:
    # if both eyes are closed, we will keep on increasing score and when eyes are open, we decrease the score
    if rpred[0] == 0 and lpred[0] == 0:
        score = score+1
        cv2.putText(frame, "Closed", (10, height-20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
    # if(rpred[0]==1 or lpred[0]==1):
    else:
        score = score-1
        cv2.putText(frame, "Open", (10, height-20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

    if score < 0:
        score = 0

    cv2.putText(frame, 'Score:' + str(score), (100, height-20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

    if score > 15:
        #person is feeling sleepy so we beep the alarm
        cv2.imwrite(os.path.join(path, 'image.jpg'), frame)
        try:
            sound.play()
        except:  # isplaying = False
            pass

        if thicc < 16:
            thicc = thicc+2
        else:
            thicc = thicc-2
            if thicc < 2:
                thicc = 2
        cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), thicc)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()


'''