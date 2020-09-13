Emotionally perceptive AI is likely going to have huge positive and negative implications on our society. Regardless of how you feel about this technology (I’m both excited and terrified), it’s inevitable and it’s important to understand how it works. Whether you’re looking to get started in building emotional AI or if you want shed some light into black box of machine learning, this is for you.

This project had two parts. The first was creating a support-vector classification model to predict facial emotion from images using facial landmarks, a 68 point coordinate system that maps onto the “moving parts” of the face. Then, this model was fed into a live video stream from a webcam, where it classifies the emotion of all present faces frame by frame, and returns a prediction in real time. See below:

![](src/visualization/emotion_stream.gif)


### Tools
Many tools were used for this project, including python, pandas, numpy, matplotlib, sklearn, keras, Dlib and openCV.

## Part I: Creating the model
I trained my data on the Cohn-Kanade CK+ dataset which can be found here. The dataset contained a few hundred pre-labeled images of various faces expressing the following emotions: anger, contempt, disgust, happy, neutral, sadness, and surprise. As the contempt and sadness categories had too few images to properly classify, they were dropped from the model.

The directory structure is a bit unwieldy, so use the code (python 3) below to get it sorted properly (this code is slightly modified from a tutorial provided by Paul Vangent , thanks Paul!).

Before running the code, two directories within the root data directory. Name one ‘sorted_set’ and the other ‘data_set’ and populate each with an empty subdirectory for each of the eight emotions.

This code organizes your images into nicely labeled subdirectories as well as capturing only the first (neutral) and last (peak labeled emotion) from each sequence.

```
import cv2
import glob
import tensorflow as tf
from shutil import copyfile
import os
import pandas as pd
import csv
 
def organize_data():
     '''This function sorts the downloaded folder structure 
     so that a subdirectory for each emotion is populated
     with their corresponding images.'''
 
    emotions = ["neutral", "anger", "contempt", "disgust", 
    "fear", "happy", "sadness", "surprise"] #Define emotion 
    order
    participants = glob.glob("source_emotion//*") #Returns a 
    list of all folders with participant numbers
     
    for x in participants:
        part = "%s" %x[-4:] #store current participant 
        number
        for sessions in glob.glob("%s//*" %x): #Store list 
        of sessions for current participant
            for files in glob.glob("%s//*" %sessions):
                current_session = files[20:-30]
 
                with open(files, 'r') as f:
                    file = f.read()
                emotion = int(float(file)) #emotions are 
                encoded as a float, readline as float, then 
                convert to integer.
 
                sourcefile_emotion = 
    sorted(glob.glob("source_images/%s/%s/*" %(part, 
    current_session)))[-1] #get path for last image in 
    sequence, which contains the emotion
                sourcefile_neutral = 
    sorted(glob.glob("source_images/%s/%s/*" %(part, 
    current_session)))[0] #do same for neutral image
                       
                dest_neut = "sorted_set//neutral//%s" 
                %sourcefile_neutral[25:] #Generate path to 
                put neutral image
                dest_emot = "sorted_set//%s//%s" % . 
                (emotions[emotion], sourcefile_emotion[25:]) 
                #Do same for emotion containing image
 
                copyfile(sourcefile_neutral, dest_neut) 
                #Copy file
                copyfile(sourcefile_emotion, dest_emot) 
                #Copy file
 
organize_data()
```

Next, run this code, which finds faces in the images, crops them to reduce noise, converts them to grayscale, and tucks them neatly away for modeling in your ‘data_set’ directory. First, download the 4 face cascade classifiers listed below and place them in your project directory.

```
 faceDet = cv2.CascadeClassifier('/your_path/opencv/data/haarcascades/haarcascade_frontalface_default.xml')
 faceDet_two = cv2.CascadeClassifier("/your_path/opencv/data/haarcascades/haarcascade_frontalface_alt2.xml")
 faceDet_three = cv2.CascadeClassifier("/your_path/opencv/data/haarcascades/haarcascade_frontalface_alt.xml")
 faceDet_four = cv2.CascadeClassifier("//your_path/opencv/data/haarcascades/haarcascade_frontalface_alt_tree.xml")
 
 
def detect_faces(emotion):
 
    emotions = ["neutral", "anger", "contempt", "disgust", 
    "fear", "happy", "sadness", "surprise"] #Define emotions
     
    files = glob.glob("sorted_set/%s/*" %emotion) #Get list 
    of all images with emotion
    filenumber = 0
 
    for f in files:
        frame = cv2.imread(f) #Open image
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
    
        #Detect face using 4 different classifiers
        face = faceDet.detectMultiScale(gray, 
        scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), 
        flags=cv2.CASCADE_SCALE_IMAGE)
        face_two = faceDet_two.detectMultiScale(gray, 
        scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), 
        flags=cv2.CASCADE_SCALE_IMAGE)
        face_three = faceDet_three.detectMultiScale(gray, 
        scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), 
        flags=cv2.CASCADE_SCALE_IMAGE)
        face_four = faceDet_four.detectMultiScale(gray, 
        scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), 
        flags=cv2.CASCADE_SCALE_IMAGE)
 
        #Go over detected faces, stop at first detected 
        face, return empty if no face.
        if len(face) == 1:
            facefeatures = face
        elif len(face_two) == 1:
           facefeatures = face_two
        elif len(face_three) == 1:
            facefeatures = face_three
        elif len(face_four) == 1:
            facefeatures = face_four
        else:
            facefeatures = ""
         
        #Cut and save face
        for (x, y, w, h) in facefeatures: #get coordinates 
        and size of rectangle containing face
            print("face found in file: %s" %f)
            gray = gray[y:y+h, x:x+w] #Cut the frame to size
            try:
                out = cv2.resize(gray, (350, 350)) #Resize 
                face so all images have same size
                cv2.imwrite("data_set/%s/%s.jpg" %(emotion, 
                filenumber), out) #Write image
            except:
                pass #If error, pass file
        filenumber += 1 #Increment image number
 
for emotion in emotions:
     detect_faces(emotion)
     
```


The framed face look like this.

<a href="https://postimg.cc/F731hxxJ" target="_blank"><img src="https://i.postimg.cc/kXhSF16w/Screen-Shot-2020-09-13-at-2-04-29-PM.png" alt="Screen-Shot-2020-09-13-at-2-04-29-PM"/></a><br/><br/>


Now that you have all of your data processed and ready for modeling, create the training/testing set. It is important to reserve a portion (20%) outside of model training to test on, that way the model doesn’t just memorize the entire dataset. Before running the code, download the facial landmark algorithm and place it in your project directory.

First, set up your imports, clahe (normalizer for image pixels), facial landmark algorithm, and support-vector classifier model.

```
import cv2
import glob
import random
import math
import numpy as np
import dlib
import itertools
from sklearn.svm import SVC
import pickle
 
 
emotions = ["anger", "contempt", "disgust", "fear", "happiness", "neutral", "sadness", "surprise"] #Emotion list
 
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
detector = dlib.get_frontal_face_detector()
 
predictor = dlib.shape_predictor("/Users/cmeaton/Documents/code/ds/METIS/sea19_ds7_workingdir/project_5/src/models/shape_predictor_68_face_landmarks.dat")
 
clf = SVC(kernel='linear', probability=True, tol=1e-3)#, 
verbose = True) #Set the classifier as a support vector machines with polynomial kernel
 
data = {} #Make dictionary for all values
data['landmarks_vectorised'] = []
```

Now, use the following code to create your training and testing data.

```
def get_files(emotion):
    '''Define function to get file list, randomly shuffle it 
    and split 80/20'''
 
    files = glob.glob("data_set/%s/*" %emotion)
    random.shuffle(files)
    training = files[:int(len(files)*0.8)] #get first 80% of 
    file list
    prediction = files[-int(len(files)*0.2):] #get last 20% 
    of file list
    return training, prediction
```    

With your training and testing data created, it’s time to find facial landmarks for each image. The code below not only finds facial landmarks, but engineers a new feature which is the relative distance of each point to the mean, roughly the tip of the nose. Adding this feature improves overall accuracy of the model, thanks Paul.


Finding facial landmarks looks like this.

<a href="https://postimg.cc/MMnKN0TH" target="_blank"><img src="https://i.postimg.cc/Hx2cJB1w/Screen-Shot-2020-09-13-at-2-04-36-PM.png" alt="Screen-Shot-2020-09-13-at-2-04-36-PM"/></a><br/><br/>

```
def get_landmarks(image):
    '''This function locates facial landmarks and computes 
    the relative distance from the mean for each point.'''
 
    detections = detector(image, 1)
    for k,d in enumerate(detections): #For all detected face 
    instances individually
        shape = predictor(image, d) #Draw Facial Landmarks 
        with the predictor class
        xlist = []
        ylist = []
        for i in range(1,68): #Store X and Y coordinates in 
        two lists
            xlist.append(float(shape.part(i).x))
            ylist.append(float(shape.part(i).y))
        xmean = np.mean(xlist)
        ymean = np.mean(ylist)
        xcentral = [(x-xmean) for x in xlist]
        ycentral = [(y-ymean) for y in ylist]
        landmarks_vectorised = []
        for x, y, w, z in zip(xcentral, ycentral, xlist, 
        ylist):
            landmarks_vectorised.append(w)
            landmarks_vectorised.append(z)
            meannp = np.asarray((ymean,xmean))
            coornp = np.asarray((z,w))
            dist = np.linalg.norm(coornp-meannp)
            landmarks_vectorised.append(dist)
            landmarks_vectorised.append((math.atan2(y, 
            x)*360)/(2*math.pi))
        data['landmarks_vectorised'] = landmarks_vectorised
    if len(detections) < 1:
        data['landmarks_vestorised'] = "error"
```        
     
With our training and testing data separated and our facial landmark function defined, use this code to sort the data into a proper format to feed into the model, as well as perform some image processing on the way.

```
def make_sets():
    '''This function creates test/train data and labels.'''
 
    training_data = []
    training_labels = []
    prediction_data = []
    prediction_labels = []
    for emotion in emotions:
        print(" working on %s" %emotion)
        training, prediction = get_files(emotion)
        #Append data to training and prediction list, and 
        generate labels 0-7
        for item in training:
            image = cv2.imread(item) #open image
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
            #convert to grayscale
            clahe_image = clahe.apply(gray)
            get_landmarks(clahe_image)
            if data['landmarks_vectorised'] == "error":
                print("no face detected on this one")
            else:
                 
    training_data.append(data['landmarks_vectorised']) 
    #append image array to training data list
                 
    training_labels.append(emotions.index(emotion))
        for item in prediction:
            image = cv2.imread(item)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            clahe_image = clahe.apply(gray)
            get_landmarks(clahe_image)
            if data['landmarks_vectorised'] == "error":
                print("no face detected on this one")
            else:       
          
    prediction_data.append(data['landmarks_vectorised'])             
    prediction_labels.append(emotions.index(emotion))
    return training_data, training_labels, 
    prediction_data, prediction_labels
```

Finally, everything is in place to train the support-vector classification model, which is done by the code below. It performed with 82% accuracy.

```
accur_lin = []
for i in range(0,10):
    print("Making sets %s" %i) #Make sets by random sampling 
    80/20%
 
    training_data, training_labels, prediction_data, 
    prediction_labels = make_sets()
    npar_train = np.array(training_data) #Turn the training 
    set into a numpy array for the classifier
    npar_trainlabs = np.array(training_labels)
    print("training SVM linear %s" %i) #train SVM
    clf.fit(npar_train, training_labels)
    print("getting accuracies %s" %i) #Use score() function 
    to get accuracy
     
    npar_pred = np.array(prediction_data)
    pred_lin = clf.score(npar_pred, prediction_labels)
    print(f"linear: {pred_lin}")
    accur_lin.append(pred_lin) #Store accuracy in a list
    print("Mean value lin svm: %s" %np.mean(accur_lin)) 
    #Get mean accuracy of the 10 runs
 
filename = 'your_path.sav'
pickle.dump(clf, open(filename, 'wb'))

```

## Part II: Live stream predicting
With the model pickled, it is time to create the live stream predictor from your webcam. This first bit of code you’ve seen before and you know how it works. Just copy it into your notebook or .py file and move to the next portion.

```
import cv2
import numpy as np
from imutils import face_utils
import glob
import random
import math
import dlib
import itertools
from sklearn.svm import SVC
import pickle
 
# load model
filename = '/Users/cmeaton/Documents/code/ds/METIS/sea19_ds7_workingdir/project_5/src/models/saved_models/lin_svm_model.sav'
loaded_model = pickle.load(open(filename, 'rb'))
 
# load facial landmark algorithms
p = '/Users/cmeaton/Documents/code/ds/METIS/sea19_ds7_workingdir/project_5/src/models/saved_models/face_algorithms/shape_predictor_68_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)
 
#load facial finder algorithms
faceDet = cv2.CascadeClassifier(
    '/Users/cmeaton/Documents/code/ds/METIS/sea19_ds7_workingdir/project_5/src/models/saved_models/face_algorithms/haarcascade_frontalface_default.xml')
 
faceDet_two = cv2.CascadeClassifier(    "/Users/cmeaton/Documents/code/ds/METIS/sea19_ds7_workingdir/project_5/src/models/saved_models/face_algorithms//haarcascade_frontalface_alt2.xml")
 
faceDet_three = cv2.CascadeClassifier(    "/Users/cmeaton/Documents/code/ds/METIS/sea19_ds7_workingdir/project_5/src/models/saved_models/face_algorithms//haarcascade_frontalface_alt.xml")
 
faceDet_four = cv2.CascadeClassifier(   "/Users/cmeaton/Documents/code/ds/METIS/sea19_ds7_workingdir/project_5/src/models/saved_models/face_algorithms//haarcascade_frontalface_alt_tree.xml")
 
# placeholder for data to be used later
data = {}
# normalization of image arrays
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
 
def get_landmarks(image):
    '''This function locates facial landmarks and computes 
    the relative distance from the mean for each point.'''
 
 
    training_data = []
    detections = detector(image, 1)
    for k,d in enumerate(detections): #For all detected face 
    instances individually
        shape = predictor(image, d) #Draw Facial Landmarks 
        with the predictor class
        xlist = []
        ylist = []
        for i in range(1,68): #Store X and Y coordinates in 
        two lists
            xlist.append(float(shape.part(i).x))
            ylist.append(float(shape.part(i).y))
        xmean = np.mean(xlist)
        ymean = np.mean(ylist)
        xcentral = [(x-xmean) for x in xlist]
        ycentral = [(y-ymean) for y in ylist]
        landmarks_vectorised = []
        for x, y, w, z in zip(xcentral, ycentral, xlist, 
        ylist):
            landmarks_vectorised.append(w)
            landmarks_vectorised.append(z)
            meannp = np.asarray((ymean,xmean))
            coornp = np.asarray((z,w))
            dist = np.linalg.norm(coornp-meannp)
            landmarks_vectorised.append(dist)
            landmarks_vectorised.append((math.atan2(y, 
            x)*360)/(2*math.pi))
        data['landmarks_vectorised'] = landmarks_vectorised
    if len(detections) < 1:
        data['landmarks_vestorised'] = "error"
 
    training_data.append(data['landmarks_vectorised'])
    return training_data
 
def crop_face(imgarray, section, margin=40, size=64):
    """
    :param imgarray: full image
    :param section: face detected area (x, y, w, h)
    :param margin: add some margin to the face detected area to include a full head
    :param size: the result image resolution with be (size x size)
    :return: resized image in numpy array with shape (size x size x 3)
    """
 
    img_h, img_w, _ = imgarray.shape
    if section is None:
        section = [0, 0, img_w, img_h]
    (x, y, w, h) = section
    margin = int(min(w,h) * margin / 100)
    x_a = x - margin
    y_a = y - margin
    x_b = x + w + margin
    y_b = y + h + margin
    if x_a < 0:
        x_b = min(x_b - x_a, img_w-1)
        x_a = 0
    if y_a < 0:
        y_b = min(y_b - y_a, img_h-1)
        y_a = 0
    if x_b > img_w:
        x_a = max(x_a - (x_b - img_w), 0)
        x_b = img_w
    if y_b > img_h:
        y_a = max(y_a - (y_b - img_h), 0)
        y_b = img_h
    cropped = imgarray[y_a: y_b, x_a: x_b]
    resized_img = cv2.resize(cropped, (size, size), 
    interpolation=cv2.INTER_AREA)
    resized_img = np.array(resized_img)
    return resized_img, (x_a, y_a, x_b - x_a, y_b - y_a)
```

Below the code above, paste this portion. This portion fires up your webcam, does the preprocessing we previously defined, draws a frame around each found face, and uses our model to predict the emotion.

```
video_capture = cv2.VideoCapture(0)
while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceDet_three.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(224, 224),
   )
 
    face_imgs = np.empty((len(faces), 350, 350, 3))
 
    for i, face in enumerate(faces):
        face_img, cropped = crop_face(frame, face, 
        margin=40, size=350)
        (x, y, w, h) = cropped
        cv2.rectangle(frame, (x, y), (x + w, y + h), (350, 
        350, 3), 2)
        face_imgs[i,:,:,:] = face_img
 
        results = (get_landmarks(face_img))
        model = loaded_model.predict_proba(results)        
```

The next bit of code creates the text that is cast onto each frame, labeling the prediction. I tinkered with font/color to try to match the emotion, as well as the sensitivity of prediction. For example, the model was not very sensitive to anger or disgust, so I created a lower threshold requirement to classify these emotions. This makes the prediction labeling a little more sensitive overall, but correctly classifying anger is worth a couple of happiness misclassifications.

```
if model[0][0] > .3:
    emotion = 'Anger'
    fontColor = (255, 0, 0)
    font = cv2.FONT_ITALIC
    fontScale = 1.5
elif model[0][1] > .3:
    emotion = 'Disgust'
    fontColor = (60, 179, 113)
    font = cv2.FONT_HERSHEY_TRIPLEX
    fontScale = 1.5
elif model[0][2] > .5:
    emotion = 'Fear'
    fontColor = (255, 0, 0)
    font = cv2.FONT_HERSHEY_TRIPLEX
    fontScale = 1
elif model[0][3] > .5:
    emotion = 'Happy'
    fontColor = (0, 221, 221)
    font = cv2.FONT_HERSHEY_SCRIPT_COMPLEX
    fontScale = 2
elif model[0][4] > .4:
    emotion = 'Neutral'
    fontColor = (255,255,255)
    font = cv2.FONT_ITALIC
    fontScale = 1.5
elif model[0][5] > .3:
    emotion = 'Surprise'
    fontColor = (196, 0, 255)
    font = cv2.FONT_HERSHEY_TRIPLEX
    fontScale = 1.5
 
bottomLeftCornerOfText = (10,100)
lineType               = 2
cv2.putText(frame, f'Predicted emotion: {emotion}',
            bottomLeftCornerOfText,
            font,
            fontScale,
            fontColor,
            lineType)
            
```

This last bit of code finds and displays facial landmarks on each face in every frame. It also allows you to close the stream by pressing ‘q’.

```
    # Get faces into webcam's image
    rects = detector(gray, 0)
 
    # For each detected face, find the landmark.
    for (i, rect) in enumerate(rects):
        # Make the prediction and transfom it to numpy array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
 
        # Draw on our image, all the found coordinate points 
        (x,y)
        for (x, y) in shape:
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
    cv2.imshow('Keras Faces', frame)
 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
 
# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
```

Here is a snapshot of the model in action.

<a href="https://postimg.cc/MMnKN0TH" target="_blank"><img src="https://i.postimg.cc/Hx2cJB1w/Screen-Shot-2020-09-13-at-2-04-36-PM.png" alt="Screen-Shot-2020-09-13-at-2-04-36-PM"/></a><br/><br/>

### Future Work
As with any passion project, there is tremendous room for growth. Immediate improvements will be centered around training the model on more comprehensive data. Fortunately, AffectNet, a more recent and more robust pre-labeled dataset for facial emotion recognition, was recently released. Retraining the model on this data, which has almost half a million images will likely improve accuracy and increase the range of emotional categories.

In addition, emotion is just one target to classify from facial images. Currently, data can be found to train models that predict age, gender, direction of gaze, energy level, and so on. Such models can have a myriad of use for a wide range of applications.

### Sources

- van Gent, P. (2016). Emotion Recognition With Python, OpenCV and a Face Dataset. A tech blog about fun things with Python and embedded electronics. Retrieved from:
http://www.paulvangent.com/2016/04/01/emotion-recognition-with-python-opencv-and-a-face-dataset/
- Kanade, T., Cohn, J. F., & Tian, Y. (2000). Comprehensive database for facial expression analysis. Proceedings of the Fourth IEEE International Conference on Automatic Face and Gesture Recognition (FG’00), Grenoble, France, 46-53.
- Lucey, P., Cohn, J. F., Kanade, T., Saragih, J., Ambadar, Z., & Matthews, I. (2010). The Extended Cohn-Kanade Dataset (CK+): A complete expression dataset for action unit and emotion-specified expression. Proceedings of the Third International Workshop on CVPR for Human Communicative Behavior Analysis (CVPR4HB 2010), San Francisco, USA, 94-101.
