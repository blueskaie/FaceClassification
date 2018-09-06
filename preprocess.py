import sys
import dlib
import skimage
from skimage import io
from imutils import face_utils
import imutils
import cv2
import numpy as np
import os
from skimage import transform 
from skimage.color import rgb2gray

def facealign(img):
    predictor_path = "shape_predictor_68_face_landmarks.dat"
    detector = dlib.get_frontal_face_detector()
    sp = dlib.shape_predictor(predictor_path)

    dets = detector(img, 1)
    num_faces = len(dets)
    if num_faces == 0:
        print("Sorry, there were no faces found")
        return
    # Find the 5 face landmarks we need to do the alignment.
    faces = dlib.full_object_detections()
    for detection in dets:
        faces.append(sp(img, detection))
    # It is also possible to get a single chip
    image = dlib.get_face_chip(img, faces[0], size=50)
    return image

def facemask(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    out_face = np.zeros_like(img)

    # initialize dlib's face detector (HOG-based) and then create the facial landmark predictor
    predictor_path = "shape_predictor_68_face_landmarks.dat"
    detector = dlib.get_frontal_face_detector()
    sp = dlib.shape_predictor(predictor_path)

    # detect faces in the grayscale image
    rects = detector(gray, 1)

    shape = sp(gray, rects[0])
    shape = face_utils.shape_to_np(shape)

    #initialize mask array
    remapped_shape = np.zeros_like(shape) 
    feature_mask = np.zeros((img.shape[0], img.shape[1]))   

    # we extract the face
    # remapped_shape = face_remap(shape)
    remapped_shape = cv2.convexHull(shape)
    cv2.fillConvexPoly(feature_mask, remapped_shape[0:27], 1)
    feature_mask = feature_mask.astype(np.bool)
    out_face[feature_mask] = img[feature_mask]
    # out_face[feature_mask] = 255
    return out_face

def preprocessImage(filename):
   
    try:
        img = io.imread(filename)
        img = facealign(img)
        img = facemask(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img
    except Exception:        
        img = io.imread(filename)
        # img = transform.resize(img, (50, 50))
        # img = rgb2gray(np.array(img))
        img = cv2.resize(img, (50, 50))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img
    
    
def preprocess_data_folder(data_directory):
    print("Preprocess the Data....")
    directories = [d for d in os.listdir(data_directory) 
                   if os.path.isdir(os.path.join(data_directory, d))]
    for d in directories:
        print("Processing files in Derectorie "+str(d))
        label_directory = os.path.join(data_directory, d)
        file_names = [os.path.join(label_directory, f) 
                      for f in os.listdir(label_directory) 
                      if f.endswith(".jpg")]
        i = 0
        for f in file_names:
            i+=1
            img = preprocessImage(f)
            io.imsave(f, img)
            if i%10==0:
                print("Processed "+str(i)+" files")
    return

def main():
    if len(sys.argv) < 2:
        print("input Data directory")
        exit()
    ROOT_PATH = sys.argv[1]
    # ROOT_PATH = "FaceData2"
    train_data_directory = os.path.join(ROOT_PATH, "Training")
    test_data_directory = os.path.join(ROOT_PATH, "Testing")
    # print(train_data_directory)
    # exit()
    preprocess_data_folder(train_data_directory)
    preprocess_data_folder(test_data_directory)

if __name__ == "__main__":
    main()