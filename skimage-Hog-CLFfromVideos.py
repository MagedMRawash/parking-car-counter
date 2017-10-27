import numpy as np
import pickle
import cv2

print(cv2.__version__)

# -*- coding: utf-8 -*-
svc = pickle.load(open("saved_svc.p", 'rb'))
video_src = 'dataset/video2.avi'

hog = cv2.HOGDescriptor()
hog.setSVMDetector(np.array(svc))

cap = cv2.VideoCapture(video_src)


while True:
    ret, img = cap.read()
    if (type(img) == type(None)):
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    cars = hog.detectMultiScale(gray, 1.1, 1)
    # print(svcdata.predict(X_test[0]))
    print(str(len(cars)))

    for (x, y, w, h) in cars:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

    cv2.imshow('video', img)

    if cv2.waitKey(33) == 27:
        break

cv2.destroyAllWindows()
