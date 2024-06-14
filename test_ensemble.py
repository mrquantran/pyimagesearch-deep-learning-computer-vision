# import the necessary packages
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from keras.api.layers import load_model
from keras.api.optimizers import SGD
from keras.src.datasets import cifar10
import numpy as np
import argparse
import glob
import os

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--models", required=True, help="path to models directory")
args = vars(ap.parse_args())

# load the training and testing data, then scale it into the range [0, 1]
print("[INFO] loading CIFAR-10 data...")
((trainX, trainY), (testX, testY)) = cifar10.load_data()
trainX = trainX.astype("float") / 255.0
testX = testX.astype("float") / 255.0

# initialize the label names for the CIFAR-10 dataset
labelNames = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

# convert the labels from integers to vectors
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

# construct the path used to collect the models then initialize the models list
modelPaths = os.path.sep.join([args["models"], "*.model"])
modelPaths = list(glob.glob(modelPaths))
models = []

# loop over the model paths, loading the model, and adding it to the list of models
for (i, modelPath) in enumerate(modelPaths):
    print("[INFO] loading model {}/{}".format(i + 1, len(modelPaths)))
    models.append(load_model(modelPath))

# initialize the list of predictions
print("[INFO] evaluating ensemble...")
predictions = []

# loop over the models
for model in models:
    # make predictions on the testing data, then add these predictions to the predictions list
    predictions.append(model.predict(testX, batch_size=64))

# average the probabilities across all model predictions, then show a classification report
predictions = np.average(predictions, axis=0)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=labelNames))
