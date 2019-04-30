import numpy as np
import cv2
import argparse
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from skimage import feature
from imutils import build_montages
from imutils import paths


def quantify_images(image):
    features = feature.hog(image, orientations=9, pixels_per_cell=(10,10), cells_per_block=(2,2),
     transform_sqrt=True, block_norm='L1')
    return features

def load_split(path):
    imagePaths = list(paths.list_images(path))
    data = []
    labels = []

    for imagepath in imagePaths:
        label = imagepath.split(os.path.sep)[-2]

        image = cv2.imread(imagepath)
        image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (200,200))

        image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

        features = quantify_images(image)

        data.append(features)
        labels.append(label)


    return (np.array(data),np.array(labels))


#Command Line argument parser
ap = argparse.ArgumentParser()
ap.add_argument('-d', '--dataset', required=True, help='path to input dataset')
ap.add_argument('-t', '--trials', type=int, default=5, help='# of trials to run')
args = vars(ap.parse_args())

#define the training and testing paths
training_path = os.path.sep.join([args['dataset'], 'training'])
testing_path = os.path.sep.join([args['dataset'],'testing'])

print('[INFO] Loading data...')
#load the data into training and testing inputs and outputs
(trainX,trainY) = load_split(training_path)
(testX,testY) = load_split(testing_path)

#encode the labels as integers
le = LabelEncoder()
trainY = le.fit_transform(trainY)
testY = le.transform(testY)

#initialize our trials dictionary
trials = {}


#start the trials
for i in range(0,args['trials']):

    #train the model
    print('Training the model {} of {}...'.format(i+1,args['trials']))

    model = RandomForestClassifier(n_estimators = 100)
    model.fit(trainX,trainY)


    #making predictions on the test dataset
    predictions = model.predict(testX)
    metrics = {}


    #Compute the confusion_matrix to measure accuracy
    cm = confusion_matrix(testY,predictions).flatten()
    (TrueNeg,FalsePos,FalseNeg,TruePos) = cm
    metrics['accuracy'] = (TruePos + TrueNeg)/float(cm.sum())
    metrics['Sensitivity'] = TruePos/float(TruePos + FalseNeg)
    metrics['Specificity'] = TrueNeg / float(TrueNeg + FalsePos)

    for (k,v) in metrics.items():
        l = trials.get(k,[])
        l.append(v)
        trials[k] = l


for metric in ('accuracy','Sensitivity','Specificity'):
    values = trials[metric]
    mean = np.mean(values)
    std = np.std(values)

    #print the computed metric
    print(metric)
    print('=' *len(metric))
    print('u={:.4f} , o={:.4f}'.format(mean,std))
    print("")


testing_paths = list(paths.list_images(testing_path))
idx = np.arange(0,len(testing_paths))
idx = np.random.choice(idx,size=(25,),replace=False)
images = []

for i in idx:
    image = cv2.imread(testing_paths[i])
    output=image.copy()
    output = cv2.resize(output,(128,128))

    image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image,(200,200))
    image = cv2.threshold(image,0,255,cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)[1]

    features = quantify_images(image)
    pred = model.predict([features])
    label = le.inverse_transform(pred)[0]

    color = (0,255,0) if label =='healthy' else(0,0,255)
    cv2.putText(output,label,(3,20),cv2.FONT_HERSHEY_SIMPLEX,0.5,color,2)
    images.append(output)

montage = build_montages(images,(128,128),(5,5))[0]


def predict_new(path):
    img = cv2.imread(path)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = cv2.resize(gray,(200,200))
    out =img.copy()
    img = cv2.threshold(img,0,255,cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)[1]
    feature = quantify_images(img)
    preds = model.predict([feature])
    lab = le.inverse_transform(preds)[0]
    Color = (0,255,0) if lab == 'healthy' else(0,0,255)
    cv2.putText(out,lab,(3,20),cv2.FONT_HERSHEY_SIMPLEX,0.5,Color,2)

    return out
V11PO03 = '/media/HassanAlsamahi/DaTa/hassan-work/Machine-Learning/Biomedical-project/Biomedical-Detect-Parkinsons/New/V11PO03.png'
path = '/media/HassanAlsamahi/DaTa/hassan-work/Machine-Learning/Biomedical-project/Biomedical-Detect-Parkinsons/New/V10PO02.png'
out = predict_new(path)

cv2.imshow("image",out)

cv2.imshow('Output',montage)
cv2.waitKey(0)
