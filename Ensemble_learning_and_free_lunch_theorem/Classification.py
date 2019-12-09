# required packages
from sklearn.cluster import KMeans
import pickle
import gzip
import numpy as np
import pandas as pd
import csv
import math
import matplotlib.pyplot
from matplotlib import pyplot as plt
from PIL import Image
import os
import numpy as np

# loss function
def getLoss(w,x,y,lam):
    m = x.shape[0]
    y_enc = onehot_enc(y) #one-hot vector notation
    a = np.dot(x,w)
    prob = softmax(a)
    loss = (-1 / m) * np.sum(y_enc * np.log(prob)) + (lam/2)*np.sum(w*w)
    grad = (-1 / m) * np.dot(x.T,(y_enc - prob)) + lam*w
    return loss,grad

# get one-hot notation
def onehot_enc(Y):
    return (np.arange(np.max(Y) + 1) == Y[:, None]).astype(float)

# soft-max function
def softmax(z):
    z -= np.max(z)
    sm = (np.exp(z).T / np.sum(np.exp(z),axis=1)).T
    return sm

#get probabilities and predictions
def getProbsAndPreds(X):
    probs = softmax(np.dot(X,w))
    preds = np.argmax(probs,axis=1)
    return probs,preds

# Accuracy function
def getAccuracy(X,Y):
    prob,prede = getProbsAndPreds(X)
    accuracy = sum(prede == Y)/(float(len(Y)))
    accuracy = accuracy*100
    return accuracy

# get confusion matrix
def compute_confusion_matrix(true, pred):
    prob,prede = getProbsAndPreds(pred)
    K = len(np.unique(true)) # Number of classes 
    result = np.zeros((K, K))
    for i in range(len(true)):
        result[true[i]][prede[i]] += 1
    return result


filename = 'mnist.pkl.gz'    # load MNIST dataset 
f = gzip.open(filename, 'rb')
training_data, validation_data, test_data = pickle.load(f, encoding='latin1')
train_x, train_y = training_data
TrainingTarget=np.array(train_y)
print(TrainingTarget.shape)
TrainingData=np.array(train_x)
print(TrainingData.shape)
validation_x, validation_y = validation_data
ValidationTarget=np.array(validation_y)
print(ValidationTarget.shape)
ValidationData=np.array(validation_x)
print(ValidationData.shape)
test_x, test_y = test_data
TestTarget=np.array(test_y)
print(TestTarget.shape)
TestData=np.array(test_x)
print(TestData.shape)


USPSMat  = []
USPSTar  = []
curPath  = 'USPSdata/Numerals'
savedImg = []

for j in range(0,10):
    curFolderPath = curPath + '/' + str(j)
    imgs =  os.listdir(curFolderPath)
    for img in imgs:
        curImg = curFolderPath + '/' + img
        if curImg[-3:] == 'png':
            img = Image.open(curImg,'r')
            img = img.resize((28, 28))
            savedImg = img
            imgdata = (255-np.array(img.getdata()))/255
            USPSMat.append(imgdata)
            USPSTar.append(j)
USPSTar= np.asarray(USPSTar)
USPSMat= np.asarray(USPSMat)            
print(USPSMat.shape)
print(USPSTar.shape)

ErmsArr = []
AccuracyArr = []
num_classes = 10

w = np.zeros([TrainingData.shape[1],len(np.unique(TrainingTarget))])
lam = 1
iterations = 20
learningRate = 0.01
losses = []
for i in range(0,iterations):
    loss,grad = getLoss(w,TrainingData,TrainingTarget,lam)
    losses.append(loss)
    w = w - (learningRate * grad)
print (loss)
print ('Training Accuracy: ', getAccuracy(TrainingData,TrainingTarget))
print ('Test Accuracy: ', getAccuracy(TestData,TestTarget))
print ('Test Accuracy: ', getAccuracy(USPSMat,USPSTar))
cm1 = compute_confusion_matrix(TrainingTarget, TrainingData)
print(cm1)
cm2= compute_confusion_matrix(USPSTar, USPSMat)
print(cm2)


# Neural Network

import keras
from keras.datasets import mnist
from keras.layers import Dense
from keras.models import Sequential

(x_train, y_train), (x_test, y_test) = mnist.load_data()
num_classes=10
image_vector_size=28*28
x_train = x_train.reshape(x_train.shape[0], image_vector_size)
x_test = x_test.reshape(x_test.shape[0], image_vector_size)
USPSMat = USPSMat.reshape(USPSMat.shape[0], image_vector_size)
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
USPSTar = keras.utils.to_categorical(USPSTar, num_classes)
image_size = 784 
model = Sequential()
model.add(Dense(units=32, activation='sigmoid', input_shape=(image_size,)))
model.add(Dense(units=num_classes, activation='softmax'))
model.compile(optimizer='sgd', loss='categorical_crossentropy',metrics=['accuracy'])
history = model.fit(x_train, y_train, batch_size=128, epochs=100,
verbose=False,validation_split=.1)
print(x_test.shape)
print(y_test.shape)
loss,accuracy = model.evaluate(x_test, y_test, verbose=False)
print(accuracy)
loss,accuracy = model.evaluate(USPSMat,USPSTar, verbose=False)
print(accuracy)
cm3= compute_confusion_matrix(y_test, x_test)
print(cm3)
cm4= compute_confusion_matrix(USPSTar, USPSMat)
print(cm4)


# SVM & RandomForest

import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import fetch_mldata

mnist = fetch_mldata('MNIST original')
n_train = 60000
n_test = 10000
indices = np.arange(len(mnist.data))
train_idx = np.arange(0,n_train)
test_idx = np.arange(n_train+1,n_train+n_test)
X_train, y_train = mnist.data[train_idx], mnist.target[train_idx]
X_test, y_test = mnist.data[test_idx], mnist.target[test_idx]
# SVM
classifier1 = SVC(kernel='rbf', C=2, gamma = 0.05);
classifier1.fit(X_train, y_train)
classifier1.score(X_train,y_train)
cm5= compute_confusion_matrix(USPSTar, USPSMat)
print(cm5)
classifier1.fit(USPSMat, USPSTar)
classifier1.score(USPSMat, USPSTar)
cm6= compute_confusion_matrix(USPSTar, USPSMat)
print(cm6)

#RandomForestClassifier

classifier2 = RandomForestClassifier(n_estimator=10);
classifier2.fit(X_train, y_train) 
classifier2.score(X_train,y_train)
cm7= compute_confusion_matrix(USPSTar, USPSMat)
print(cm7)
classifier2.fit(USPSMat, USPSTar)
classifier2.score(USPSMat, USPSTar)
cm8= compute_confusion_matrix(USPSTar, USPSMat)
print(cm8)

# Ensemble Learning

from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
estimators = []
model1 = LogisticRegression()
estimators.append(('logistic', model1))
estimators.append(('svm', classifier1))
estimators.append(('random', classifier2))
ensemble = VotingClassifier(estimators)
results = model_selection.cross_val_score(ensemble, TrainingData, TrainingTarget, cv=kfold)
print(results.mean())
results = model_selection.cross_val_score(ensemble, USPSMat, USPSTar, cv=kfold)
print(results.mean())
