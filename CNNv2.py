from keras.models import Sequential, Model
from keras.layers import  Convolution2D, Dropout, Conv2D, BatchNormalization, GlobalAveragePooling2D,MaxPooling2D, Flatten, Dense
from keras.applications.xception import Xception
from keras_preprocessing.image import ImageDataGenerator
from keras.utils import plot_model, np_utils

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score,confusion_matrix
from sklearn import datasets

import keras_metrics as km
import matplotlib.pyplot as plt
import numpy as np
import os
import split_folders
import shutil

seedNumbers = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
batchSize = 32
test_dir_name = '/Users/haikristianlethanh/Desktop/test/val'
train_dir_name = '/Users/haikristianlethanh/Desktop/test/train'
path_to_dataset = '/Users/haikristianlethanh/Desktop/FIRST_Data'
splited_dir_path = '/Users/haikristianlethanh/Desktop/test'

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
    numfiles
    Y_pred = classifier.predict_generator(test_set,steps=(numfiles // 32) + 1)
    Y_pred.shape
    classes = test_set.classes[test_set.index_array]
    classes
    test_set.classes
    y_pred = np.argmax(Y_pred, axis=-1)
    print("Accuracy score: ", accuracy_score(y_pred, test_set.classes))
    y_pred
    cm = confusion_matrix(test_set.classes, y_pred)
    print(cm)
    print(classification_report(test_set.classes, y_pred,target_names=test_set.class_indices.keys()))
    
def initializeModel():
    ##zjednodusit neuronku 2x konvulencne, 1x dropout
    classifier.add(Conv2D(32, (3, 3), input_shape=(64,64,3), activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(2,2)))
    classifier.add(Conv2D(64, (3, 3),input_shape=(64,64,3), activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(2,2)))
    classifier.add(Dropout(0.2))
    classifier.add(Flatten())
    classifier.add(Dense(units = 194, activation='relu'))
    classifier.add(Dense(units = 4, activation='softmax'))
    recall = km.binary_recall(label=0)
    precision = km.binary_precision(label=1)
    c_precision = km.categorical_precision()
    classifier.compile(optimizer= 'Adam', loss = 'categorical_crossentropy', metrics = [c_precision, 'accuracy', recall, precision])   
    classifier.summary()
    classifier.metrics_names



for i in seedNumbers:
    ## split dataset
    print(i)
    split_folders.ratio(path_to_dataset, output=splited_dir_path, seed=i, ratio=(.8, .2)) # default values
    print('Splitted')
    ## init model
    classifier =  Sequential()
    initializeModel()
# =============================================================================
#     ## load dataset CMD+Shift+4
# =============================================================================
    ## train_datagen = ImageDataGenerator(width_shift_range=[0,0.3])
    ##train_datagen = ImageDataGenerator(height_shift_range=0.40, width_shift_range=[0,0.3])
    ##train_datagen = ImageDataGenerator(horizontal_flip=True, vertical_flip=True, height_shift_range=0.40, width_shift_range=[0,0.3])
    ##train_datagen = ImageDataGenerator(brightness_range=[1,1.5], horizontal_flip=True, vertical_flip=True, rotation_range=360)
    train_datagen = ImageDataGenerator(brightness_range=[1,1.5], horizontal_flip=True, vertical_flip=True, zoom_range=[0.7,1.0])
    test_datagen = ImageDataGenerator()
    training_set = train_datagen.flow_from_directory(train_dir_name,target_size=(64,64),batch_size=batchSize,class_mode='categorical')
    test_set = test_datagen.flow_from_directory(test_dir_name, target_size= (64,64), batch_size=batchSize, class_mode='categorical', shuffle=False)
# =============================================================================
#     ## train model
# =============================================================================
    numfiles = sum([len(files) for r, d, files in os.walk(train_dir_name)])
    history = classifier.fit_generator(training_set, steps_per_epoch=numfiles/batchSize, nb_epoch=20)   
# =============================================================================
#     ## summary
# =============================================================================
    trainingLoss()
    metrics()
# =============================================================================
#     ## remove splited dataset
# =============================================================================
    shutil.rmtree(splited_dir_path)
    print('Removed')


 



