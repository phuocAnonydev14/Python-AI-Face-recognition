import cv2
import face_recognition
import os
from datetime import datetime
import numpy as np
from deepface import DeepFace

path = 'pic2'
images = []
classNames = []

# dir images
myList = os.listdir(path)
print(myList)

for cl in myList:
    # read image
    curImg = cv2.imread(os.path.join(path, cl))
    images.append(curImg)
    # split text to separate by the dot in path
    # get name file except extension
    classNames.append(os.path.splitext(cl)[0])

print(len(images))


#step encoding
def encode(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodeList.append(face_recognition.face_encodings(img)[0])
    return encodeList


encodeListKnow = encode(images)


def attend(name, age, gender):
    with open("data.csv", "r+") as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dateTimeString = now.strftime("%H:%M:%S")
            f.writelines(f"\n{name},{age},{gender},{dateTimeString}")


# start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    frameS = cv2.resize(frame, (0, 0), None, 0.5, 0.5)
    frameS = cv2.cvtColor(frameS, cv2.COLOR_BGR2RGB)

    # determine face location
    faceCurFrame = face_recognition.face_locations(frameS)  # lay tung khuon mat va vi tri khuon mat hien tai
    encodeCurFrame = face_recognition.face_encodings(frameS)

    for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFrame):  # get each face and faceLoc to show
        matches = face_recognition.compare_faces(encodeListKnow, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnow, encodeFace)
        matchIndex = np.argmin(faceDis)  # get faceDis min

        if faceDis[matchIndex] < 0.5:
            # get name & uppercase
            name = classNames[matchIndex].upper()
            # Predict age and gender
            try:
                analysis = DeepFace.analyze(frame, actions=['age', 'gender'], enforce_detection=True)
                print(analysis)
                age = analysis[0]['age']
                gender = analysis[0]['dominant_gender']
            except:
                age = "Unknown"
                gender = "Unknown"
            attend(name, age, gender)
        else:
            name = "Unknown"
            age = "Unknown"
            gender = "Unknown"

        # print name
        y1, x2, y2, x1 = faceLoc
        y1, x2, y2, x1 = y1 * 2, x2 * 2, y2 * 2, x1 * 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, f"{name} {age} {gender}", (x2, y2), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 255), 2)

    cv2.imshow('phuoc show', frame)
    if cv2.waitKey(1) == ord("q"):  # enter q will exit
        break

cap.release()  # giai phong camera
cv2.destroyAllWindows()  # exit all window
