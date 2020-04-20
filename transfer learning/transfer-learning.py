from keras.models import Sequential, Model
from keras.layers import  Convolution2D, Dropout, Conv2D, BatchNormalization, GlobalAveragePooling2D,MaxPooling2D, Flatten, Dense
from keras.applications.xception import Xception
from keras_preprocessing.image import ImageDataGenerator
from keras.utils import plot_model, np_utils
from keras.applications.resnet50 import ResNet50

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score,confusion_matrix
from sklearn import datasets

import keras_metrics as km
import matplotlib.pyplot as plt
import numpy as np
import os
import split_folders
import shutil
import keras

seedNumbers = [2]
path_to_dataset = '/Users/haikristianlethanh/Desktop/CNN-keras-deep-learning/FIRST_Data'
splited_dir_path = "/Users/haikristianlethanh/Desktop/test"
test_dir_name = '/Users/haikristianlethanh/Desktop/test/val'
train_dir_name = '/Users/haikristianlethanh/Desktop/test/train' 
logFile= 'logs.txt'


for i in seedNumbers:
    split_folders.ratio(path_to_dataset, output=splited_dir_path, seed=i, ratio=(.8, .2)) # default values
    restnet=ResNet50(weights = 'imagenet',include_top=True)
    classifier = Sequential()
    classifier.add(restnet)
    classifier.add(Dense(4, activation='softmax'))
    classifier.compile(optimizer= 'Adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])  
    train_datagen = ImageDataGenerator()
    test_datagen = ImageDataGenerator()
    training_set = train_datagen.flow_from_directory('/Users/haikristianlethanh/Desktop/test/train',target_size=(224,224),batch_size=32,class_mode='categorical')
    test_set = test_datagen.flow_from_directory('/Users/haikristianlethanh/Desktop/test/val', target_size= (224,224), batch_size=32, class_mode='categorical', shuffle=False)
    numfiles = sum([len(files) for r, d, files in os.walk(train_dir_name)])
    history = classifier.fit_generator(training_set, steps_per_epoch=numfiles/32, nb_epoch=10) 
    numfiles = sum([len(files) for r, d, files in os.walk(test_dir_name)])
    Y_pred = classifier.predict_generator(test_set,steps=(numfiles // 32) + 1)
    classes = test_set.classes[test_set.index_array]
    classes
    test_set.classes
    y_pred = np.argmax(Y_pred, axis=-1)    
    accuracy = accuracy_score(y_pred, test_set.classes)
    print("Accuracy score: ", accuracy)
    f = open(logFile, 'a+')
    text=" Accuracy score: "+ str(accuracy)
    f.write(text)
    f.close()
    y_pred
    cm = confusion_matrix(test_set.classes, y_pred)
    print(cm)
    f = open(logFile, 'a+')
    text= "\n" + str(cm)
    f.write(text)
    f.close()
    print(classification_report(test_set.classes, y_pred,target_names=test_set.class_indices.keys()))
    f = open(logFile, 'a+')
    text="\n" + str(classification_report(test_set.classes, y_pred,target_names=test_set.class_indices.keys()))
    f.write(text)
    f.close()
    ## remove splited dataset
    shutil.rmtree('/Users/haikristianlethanh/Desktop/test')
    print('Removed')


 



