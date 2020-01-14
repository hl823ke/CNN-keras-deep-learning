from keras.models import Sequential, Model
from keras.layers import  Convolution2D, Dropout, Conv2D, BatchNormalization, GlobalAveragePooling2D,MaxPooling2D, Flatten, Dense
from keras.applications.xception import Xception
from keras_preprocessing.image import ImageDataGenerator
from keras.utils import plot_model, np_utils

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score,confusion_matrix
from sklearn import datasets

from PIL import Image
from scipy.ndimage import zoom

import keras_metrics as km
import matplotlib.pyplot as plt
import numpy as np
import os
import split_folders
import shutil

import cv2
import glob, os, errno
import Preprocess



seedNumbers = [2]
batchSize = 32
##test_dir_name = '/Users/haikristianlethanh/Desktop/test/val'
test_dir_name = '/Users/haikristianlethanh/Desktop/CNN-keras-deep-learning/GREY'
train_dir_name = '/Users/haikristianlethanh/Desktop/test/train'
path_to_dataset = '/Users/haikristianlethanh/Desktop/FIRST_Data'
splited_dir_path = '/Users/haikristianlethanh/Desktop/test'

def clipped_zoom(img, zoom_factor, **kwargs):
    h, w = img.shape[:2]
    # For multichannel images we don't want to apply the zoom factor to the RGB
    # dimension, so instead we create a tuple of zoom factors, one per array
    # dimension, with 1's for any trailing dimensions after the width and height.
    zoom_tuple = (zoom_factor,) * 2 + (1,) * (img.ndim - 2)

    # Zooming out
    if zoom_factor < 1:

        # Bounding box of the zoomed-out image within the output array
        zh = int(np.round(h * zoom_factor))
        zw = int(np.round(w * zoom_factor))
        top = (h - zh) // 2
        left = (w - zw) // 2

        # Zero-padding
        out = np.zeros_like(img)
        out[top:top+zh, left:left+zw] = zoom(img, zoom_tuple, **kwargs)

    # Zooming in
    elif zoom_factor > 1:

        # Bounding box of the zoomed-in region within the input array
        zh = int(np.round(h / zoom_factor))
        zw = int(np.round(w / zoom_factor))
        top = (h - zh) // 2
        left = (w - zw) // 2

        out = zoom(img[top:top+zh, left:left+zw], zoom_tuple, **kwargs)

        # `out` might still be slightly larger than `img` due to rounding, so
        # trim off any extra pixels at the edges
        trim_top = ((out.shape[0] - h) // 2)
        trim_left = ((out.shape[1] - w) // 2)
        out = out[trim_top:trim_top+h, trim_left:trim_left+w]

    # If zoom_factor == 1, just return the input array
    else:
        out = img
    return out

def preproces_img(dir_images_path, dest_preprocessed):
    directory = os.fsencode(dir_images_path)
    for file in os.listdir(directory):
         filename = os.fsdecode(file)
         path_to_file = path_to_dir + filename
         ##Nacitanie obrazku uz v GrayScale
         grey_img = cv2.imread(path_to_file, cv2.IMREAD_GRAYSCALE)
         #Uprava velkosti na 150x150
         dim = (150, 150)
         resized = cv2.resize(grey_img, dim, interpolation = cv2.INTER_AREA)
         ##Nazoomovanie obrazku
         zm2 = clipped_zoom(resized, 1.5)
         status = cv2.imwrite(dest_preprocessed + filename, zm2)
         continue
 

def trainingLoss(history):
    loss = history.history['loss']
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, color='red', label='Training loss')
    plt.title('Training  loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def metrics(test_set, classifier):
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
    classifier =  Sequential()
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
    return classifier


def main():   
    for i in seedNumbers:
        #split_folders.ratio(path_to_dataset, output=splited_dir_path, seed=i, ratio=(.8, .2)) # default values
        classifier = initializeModel()
        numfiles = sum([len(files) for r, d, files in os.walk(path_to_dataset)])
        
        train_datagen_1 = ImageDataGenerator(brightness_range=[1,1.5])
        train_datagen_2 = ImageDataGenerator(horizontal_flip=True)
        
        training_set_1 = train_datagen_1.flow_from_directory(path_to_dataset,target_size=(64,64),batch_size=batchSize,class_mode='categorical')
        training_set_2 = train_datagen_2.flow_from_directory(path_to_dataset,target_size=(64,64),batch_size=batchSize,class_mode='categorical')
        
        test_datagen = ImageDataGenerator()
        test_set = test_datagen.flow_from_directory(test_dir_name,
                                                    target_size= (64,64),
                                                    batch_size=batchSize,
                                                    class_mode='categorical',
                                                    shuffle=False)
        
        
    # =============================================================================
    #     ## train model
    # =============================================================================
        
       ##start = time.time()
        history = classifier.fit_generator(training_set_1, steps_per_epoch=numfiles/batchSize, nb_epoch=5)   
        history = classifier.fit_generator(training_set_2, steps_per_epoch=numfiles/batchSize, nb_epoch=5)   
        ##end = time.time()
        ##hours, rem = divmod(end-start, 3600)
        ##minutes, seconds = divmod(rem, 60)
        ##print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))
    # =============================================================================
    #     ## summary
    # =============================================================================
        trainingLoss(history)
        metrics(test_set, classifier)
    # =============================================================================
    #     ## remove splited dataset
    # =============================================================================
        ##shutil.rmtree(splited_dir_path)
        ##print('Removed')
    
    

main()


 # =============================================================================
 #     ## Spustat len pri potrebe upravit data
 #          1. cesta k priecinku s neupravenymi obrazkamy
 #          2. cielova dest upravenych snimkov. !!!!Cielovy priecinok musi byt vytvoreny
 # =============================================================================
 
  #preproces_img('/Users/haikristianlethanh/Desktop/CNN-keras-deep-learning/Test_downloaded/FRII/', '/Users/haikristianlethanh/Desktop/CNN-keras-deep-learning/GREY/FRII/')
    

