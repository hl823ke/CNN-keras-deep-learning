from keras.models import Sequential, Model
from keras.layers import  Convolution2D, Dropout, Conv2D, BatchNormalization, GlobalAveragePooling2D,MaxPooling2D, Flatten, Dense
from keras.applications.resnet50 import ResNet50
from keras_preprocessing.image import ImageDataGenerator
from keras.utils import plot_model, np_utils
from keras.applications.xception import Xception

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score,confusion_matrix
from sklearn import datasets

import keras_metrics as km
import matplotlib.pyplot as plt
import numpy as np
import os
import split_folders
import shutil

seedNumbers = [2]
test_dir_name = '/Users/haikristianlethanh/Desktop/test/val'
train_dir_name = '/Users/haikristianlethanh/Desktop/test/train'

def trainingLoss():
    loss = history.history['loss']
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, color='red', label='Training loss')
    plt.title('Training  loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    
def trainingLoss():
    loss = history.history['loss']
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, color='red', label='Training loss')
    plt.title('Training  loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def metrics():
    test_set.reset()
    numfiles = sum([len(files) for r, d, files in os.walk(test_dir_name)])
    Y_pred = classifier.predict_generator(test_set,steps=(numfiles // 32) + 1)
    classes = test_set.classes[test_set.index_array]
    classes
    test_set.classes
    y_pred = np.argmax(Y_pred, axis=-1)
    print("Accuracy score: ", accuracy_score(y_pred, test_set.classes))
    y_pred
    cm = confusion_matrix(test_set.classes, y_pred)
    print(cm)
    print(classification_report(test_set.classes, y_pred,target_names=test_set.class_indices.keys()))



## split dataset
split_folders.ratio('/Users/haikristianlethanh/Desktop/FIRST_Data', output="/Users/haikristianlethanh/Desktop/test", seed="2", ratio=(.8, .2)) # default values    
print('Splitted')
## init model
resnet =  ResNet50(weights = 'imagenet',include_top = False)  

output=GlobalAveragePooling2D()(resnet.output)
output=Dense(194, activation = 'relu')(output)
output=Dense(4, activation = 'softmax')(output)
model=Model(input = resnet.input, output = output)

recall = km.binary_recall(label=0)
precision = km.binary_precision(label=1)
c_precision = km.categorical_precision()
model.compile(optimizer= 'Adam', loss = 'categorical_crossentropy', metrics = [c_precision, 'accuracy', recall, precision])   
model.summary()
model.metrics_names
    
    
## load dataset
test_datagen = ImageDataGenerator()
train_datagen = ImageDataGenerator()
training_set = train_datagen.flow_from_directory('/Users/haikristianlethanh/Desktop/test/train',target_size=(64,64),batch_size=32,class_mode='categorical')
test_set = test_datagen.flow_from_directory('/Users/haikristianlethanh/Desktop/test/val', target_size= (64,64), batch_size=32, class_mode='categorical', shuffle=False)
## train model
numfiles = sum([len(files) for r, d, files in os.walk(train_dir_name)])
history = model.fit_generator(training_set, samples_per_epoch=numfiles, nb_epoch=1)   
## summary
trainingLoss()

test_set.reset()
numfiles = sum([len(files) for r, d, files in os.walk(test_dir_name)])
Y_pred = model.predict_generator(test_set,steps=(numfiles // 32) + 1)
classes = test_set.classes[test_set.index_array]
classes
Y_pred
test_set.classes
y_pred = np.argmax(Y_pred, axis=-1)
print("Accuracy score: ", accuracy_score(y_pred, test_set.classes))
y_pred
cm = confusion_matrix(test_set.classes, y_pred)
print(cm)
print(classification_report(test_set.classes, y_pred,target_names=test_set.class_indices.keys()))
## remove splited dataset
shutil.rmtree('/Users/haikristianlethanh/Desktop/test')
print('Removed')


 



