# Detecting-Parkinson-Disease
A research paper in 2017 by Zham et al explained how to detect parkinson disease without doing any MRI scans on the brain just by detecting the speed and pen presuure of sketching a spiral or a wave.
So by this we can automate the process and detect the disease by scanning the pattern of the patient and by using simple computer vision algorithms and machine learning we can train a model to classify between a patient and a healthy person.

<br /> You can find the paper here: https://www.frontiersin.org/articles/10.3389/fneur.2017.00435/full

# Algorithms used
I've used HOG (Histogram of oriented gradients) to extract features from the dataset and then passed these features to a random forest classifier to train the model on classifying patterns of patients and healthy drawings.


# How To Train the Model
You can train the model either on the wave dataset 
``` 
python3 parkinsons-detect.py --dataset dataset/wave
``` 
or by using the spirals drawings 
```
python3 parkinsons-detect.py --dataset/spiral
```
