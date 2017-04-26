import cv2
import dlib
import numpy
from sklearn import svm
import os
import sklearn
from sklearn.svm import SVC
import glob, random, math, numpy as np, itertools


PREDICTOR_PATH = "/Users/NikkiBayar1/Workspace/GenLife/GenLife/dlib/python_examples/shape_predictor_68_face_landmarks (1).dat"
predictor = dlib.shape_predictor(PREDICTOR_PATH)
cascade_path = "/Users/NikkiBayar1/opencv/data/haarcascades/haarcascade_frontalface_default.xml"
cascade = cv2.CascadeClassifier(cascade_path)

FACE_POINTS = list(range(17, 68))
MOUTH_POINTS = list(range(48, 61))
RIGHT_BROW_POINTS = list(range(17, 22))
LEFT_BROW_POINTS = list(range(22, 27))
RIGHT_EYE_POINTS = list(range(36, 42))
LEFT_EYE_POINTS = list(range(42, 48))
NOSE_POINTS = list(range(27, 35))
JAW_POINTS = list(range(0, 17))
CHIN_POINTS = list(range(6, 11))

# defining emotions
emotions = ["anger", "contempt", "disgust", "fear", "happy", "neutral", "sadness", "surprise"]

#Detects landmarks
def get_landmarks(im):
    rects = cascade.detectMultiScale(im, 1.1, 3)
    x, y, w, h = rects[0].astype(long)
    rect = dlib.rectangle(x, y, x+w, y+h)
    return numpy.array([[p.x, p.y] for p in predictor(im, rect).parts()])

# Colors landmarks 
def annotate_landmarks(im, landmarks):
    im = im.copy()
    landmarks = landmarks[FACE_POINTS]
    for idx, point in enumerate(landmarks):
        pos = (point[0, 0], point[0, 1])
        cv2.putText(im, str(idx), pos,
                    fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                    fontScale=0.4,
                    color=(0, 0, 255))
        cv2.circle(im, pos, 3, color=(0, 255, 255))
    return im

def get_images_and_labels():
    base_dir = '/Users/NikkiBayar1/Desktop/sorted_sets'
    imagesName_list = []
    label_list = []
    children = os.listdir(base_dir)
    for x in range(0, len(children)):
        children[x] = base_dir + '/' + children[x] #Ensures getting right path to image
    emotion = 0
    for folder in filter(os.path.isdir, children):  # filter on files of type directory 
        childrenImages = os.listdir(folder)
        for x in range(0, len(childrenImages)):
            if(childrenImages[x][-4:] == '.png'):
                imagesName_list.append(folder + '/' + childrenImages[x])
                label_list.append(emotion)
        emotion = emotion + 1
    return imagesName_list, label_list


# Trains the actual predictor
def make_classification(img):
    images, emotions = get_images_and_labels()
    landmarks = []   
    for image in images:
        features = get_landmarks(cv2.imread(image))
        landmarks.append(features.flatten()); #Need to flatten features to reshape data in correct way to feed into classifier
    landmarks = np.array(landmarks)    
    classifier = SVC()
    classifier.fit(landmarks, emotions)
    input_landmarks = np.array([get_landmarks(img).flatten()])
    return classifier.predict(input_landmarks)


img = cv2.imread('/Users/NikkiBayar1/Desktop/PictureData/face1.jpg')
print make_classification(img)
