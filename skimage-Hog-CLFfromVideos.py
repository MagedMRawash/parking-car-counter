import numpy as np
import pickle
import cv2
from skimage.feature import hog
from skimage.transform import pyramid_gaussian

print(cv2.__version__)
windowSize = [120,120]
stepSize = 60

def sliding_window(image, stepSize, windowSize):
    # slide a window across the image
    for x in range(0, image.shape[0], stepSize):
        for y in range(0, image.shape[1], stepSize):
            yield  (x, y, image[ x:x + windowSize[0], y:y + windowSize[0]] )


def slidPyramid(image):
    # METHOD #2: Resizing + Gaussian smoothing.
    for (i, resized) in enumerate(pyramid_gaussian( image, downscale=2)):
        # if the image is too small, break from the loop
        if resized.shape[0] < 30 or resized.shape[1] < 30:
            break
        for x, y, ImagePart in sliding_window(resized,stepSize, windowSize):
            yield (x,y, ImagePart)




####  Import Objects
svc = pickle.load(open("saved_svc.p", 'rb'))
outerImg = cv2.imread('test Data/Mijas-Car-Park.jpg')


# -*- coding: utf-8 -*-
video_src = 'test Data/video2.avi'
cap = cv2.VideoCapture(video_src)

while True:
    ret, img = cap.read()
    if (type(img) == type(None)):
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    for x, y, slice in slidPyramid(gray):
        hog_feature = hog(slice, orientations=8, pixels_per_cell=(8, 8), cells_per_block=(1, 1), visualise=False, feature_vector=True)
        length = 1800
        print(len( hog_feature))
        hog_feature = np.array([np.hstack(( hog_feature, [0] * (length - len( hog_feature)))) ])
        if svc.predict( hog_feature ) is 1 :
#             w, h = imageWindow.shape[1] ,imageWindow.shape[0]
# #             cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.imshow('video', slice)
            cv2.waitKey(0)
#
#         if cv2.waitKey(33) == 27:
#             break
#
# cv2.destroyAllWindows()
