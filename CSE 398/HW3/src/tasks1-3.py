# Computer Vision Course (CSE 40535/60535)
# University of Notre Dame
# ___________________________________________
# Andrey Kuehlkamp, Adam Czajka, November 2017

import cv2
import os
import sys
import numpy as np
from sklearn import svm
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model
import pyautogui

# *** TASK 2:
# layer of the VGG features that will be used
# cnn_codes = 'fc2'
cnn_codes = 'fc1'
clf = None

# an instance of VGG16: we need it to extract features (below)
model = VGG16(weights='imagenet')
# an alternative model, to extract features from the specified layer
# note that we could extract from any VGG16 layer, by name
features_model = Model(inputs=model.input, outputs=model.get_layer(cnn_codes).output)

# *** TASK 1 and TASK 3:
# we are going to use this list to restrict the objects our classifier will recognize
my_object_list = ['headphone','cellphone','dollar_bill','butterfly','brain','helicopter','kangaroo','elephant','chair','lotus']

def classify_svm(img):
    features = extract_vgg_features(img)
    preds = clf.predict_proba(features)
    
    # Show confidence for each class
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i in range(len(features)):
        text = ""
        for j, class_label in enumerate(clf.classes_):
            class_confidence = preds[i][j]
            text += "{}: {:.2f}".format(class_label, class_confidence)
            if j < 2:  
                text += ", "
        cv2.putText(img, text, (15, 25 + i * 25), font, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
    
    return img

def extract_vgg_features(img):
    # prepare the image for VGG
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR)
    img = img[np.newaxis, :, :, :]
    # call feature extraction
    return features_model.predict(img)


def camera_loop():
    print("Press <SPACE> to capture/classify an image, or <Esc> to exit.")
    cap = cv2.VideoCapture(0)
    while (True):
        _, frame = cap.read()

        action = cv2.waitKey(1)
        classified_frame = classify_svm(frame)
        img_to_show = cv2.resize(frame, (0,0), fx=0.8, fy=0.8)
        cv2.imshow('Camera Feed:', img_to_show)
        
        if action == ord('q') or action == 27:  # Quit if 'q' or 'Esc' key is pressed
            break
        elif action == ord(' '):  # Capture and classify if 'Space' key is pressed
            # Take a screenshot of the classified frame
            screenshot = np.zeros_like(classified_frame)
            screenshot[:] = classified_frame
            cv2.imshow('Screenshot', screenshot)
            cv2.waitKey(0)  # Wait indefinitely until a key is pressed to exit
            cv2.destroyAllWindows()
            
    cap.release()
   


if __name__ == '__main__':
    vggfile = 'vgg_features_{}.npz'.format(cnn_codes)

    # train the SVM to detect selected objects
    if os.path.exists(vggfile):
        # load pre-extracted features for all objects in Caltech 101
        print('Loading pre-extracted VGG features...')
        npzfile = np.load(vggfile)
        vgg_features = npzfile['vgg_features']
        labels = npzfile['labels']

        # filter out only the desired objects
        valid_indices = [n for n, l in enumerate(labels) if l in my_object_list]
        vgg_features = vgg_features[valid_indices]
        labels = labels[valid_indices]
    else:
        print("Pre-extracted features not found:", vggfile)
        sys.exit(0)

    # *** TASK 2:
    print("Training SVM ...")


    print(vgg_features.shape)
    print(labels.shape)

    # Train SVM classifier with probability estimates enabled
    clf = svm.SVC(kernel='linear', probability=True).fit(vgg_features, labels)
    # clf = svm.SVC(kernel='poly',probability=True,degree=3).fit(vgg_features, labels)
    # clf = svm.SVC(kernel='rbf',probability=True, gamma='auto').fit(vgg_features, labels)

    camera_loop()
    cv2.destroyAllWindows()
