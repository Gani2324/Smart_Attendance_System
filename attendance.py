
import face_recognition
import cv2
import os
import numpy as np
from datetime import datetime

def mark_attendance(name):
    with open("attendance.csv", "a") as f:
        now = datetime.now()
        dt = now.strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"{name},{dt}\n")

known_faces = []
known_names = []

for person in os.listdir("dataset"):
    for img_file in os.listdir(f"dataset/{person}"):
        img = face_recognition.load_image_file(f"dataset/{person}/{img_file}")
        encoding = face_recognition.face_encodings(img)[0]
        known_faces.append(encoding)
        known_names.append(person)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    locations = face_recognition.face_locations(rgb_frame)
    encodings = face_recognition.face_encodings(rgb_frame, locations)

    for encode, loc in zip(encodings, locations):
        matches = face_recognition.compare_faces(known_faces, encode)
        name = "Unknown"

        if True in matches:
            match_index = matches.index(True)
            name = known_names[match_index]
            mark_attendance(name)

        top, right, bottom, left = loc
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("Attendance System", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
