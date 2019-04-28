#initioalize network
from keras.models import Sequential, Model
# to add convuluion layer
from keras.layers import  Convolution2D, Dropout, Conv2D, BatchNormalization, GlobalAveragePooling2D
from keras.applications.xception import Xception
# proceed pooling layer
from keras.layers import MaxPooling2D, Conv2D
#pooling to large vector
from keras.layers import  Flatten
#used to add to full conected network
from keras.layers import Dense

from keras_preprocessing.image import ImageDataGenerator
from keras.utils import plot_model, np_utils
import matplotlib.pyplot as plt
import numpy as np
import keras_metrics as km
import split_folders

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import datasets
 
print('[INFO] loading MNIST full dataset...')
dataset = datasets.fetch_mldata("MNIST Original")
 
(trainX, testX, trainY, testY) = train_test_split(data, dataset.target, test_size=0.25)

split_folders.ratio('/Users/haikristianlethanh/Desktop/FIRST_Data', output="/Users/haikristianlethanh/Desktop/SplitedData", seed=1337, ratio=(.8, .1, .1))

#Initialize the CNN

classifier =  Sequential()

#1 Convolution
# Create many features map, 32 is common start point
# input shape = set same format
# we use rectifire activation function
classifier.add(Convolution2D(32, 3, 3, input_shape=(64,64,3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))
classifier.add(Dropout(0.2))
classifier.add(Convolution2D(64, 3, 3, input_shape=(64,64,3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))
classifier.add(Dropout(0.2))
classifier.add(Convolution2D(194, 3, 3, input_shape=(64,64,3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))
classifier.add(Dropout(0.2))


#2 Pooling
# It gets maximum value from features map and create pooled feature map, it call max pooling
# improve complexity and performance
# 2x2 we still keep it precise and dont loose ani information
#classifier.add(MaxPooling2D(pool_size = (2,2)))
#classifier.add(BatchNormalization())
#classifier.add(Dropout(0.2))

#3 Flattening
#Create vector
classifier.add(Flatten())

#4 Full connection
#hidden layer
# 128 needed experiments to decide correct number
classifier.add(Dense(output_dim = 194, activation='relu'))
classifier.add(Dropout(0.5))
classifier.add(Dense(output_dim = 4, activation='softmax'))

recall = km.binary_recall(label=0)
precision = km.binary_precision(label=1)
c_precision = km.categorical_precision()
sc_precision = km.sparse_categorical_precision()

c_precision, 'accuracy', sc_precision, recall, precision
#5 compile the CNN
classifier.compile(optimizer= 'Adam', loss = 'categorical_crossentropy', metrics = [c_precision, 'accuracy', sc_precision, recall, precision])

#Image augmentation. Do this?

train_datagen = ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip=False)

test_datagen = ImageDataGenerator(rescale=1./255)
validation_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('/Users/haikristianlethanh/Desktop/SplitedData/train',target_size=(64,64),batch_size=32,class_mode='categorical')

validation_set = validation_datagen.flow_from_directory('/Users/haikristianlethanh/Desktop/SplitedData/val',target_size=(64,64),batch_size=32,class_mode='categorical')

test_set = test_datagen.flow_from_directory('/Users/haikristianlethanh/Desktop/SplitedData/test', target_size= (64,64), batch_size=64, class_mode='categorical')
#validation_set = validation_datagen.flow_from_directory('/Users/haikristianlethanh/Desktop/DP/transfer_learning/dataset1/validation', target_size= (64,64), batch_size=32, class_mode='categorical')

history = classifier.fit_generator(
    training_set,
    samples_per_epoch = 3660,
    nb_epoch= 20,
    validation_data= validation_set,
    nb_val_samples= 915
)
classifier.summary()
classifier.metrics_names
classifier.evaluate_generator(training_set,steps=461, pickle_safe=True )

classifier.predict_generator(test_set, steps=(461 // 32) + 1)
predIdxs = classifier.predict_generator(test_set,steps=(461 // 64) + 1)
predIdxs = np.argmax(predIdxs, axis=1)
predIdxs
test_set.classes
from sklearn.metrics import accuracy_score
print("Accuracy score: ", accuracy_score(predIdxs, test_set.classes))


test_imgs, test_labels = next(test_set)
test_labels = test_labels[:,0]
test_labels
filenames = test_set.filenames
len(filenames)
nb_samples = len(filenames)
predIdxs = classifier.predict_generator(test_set,steps=(461 // 32) + 1)

test_set
test_set.class_indices.keys()
predIdxs = np.argmax(predIdxs, axis=1)

from sklearn.metrics import accuracy_score
print("Accuracy score: ", accuracy_score(predIdxs, test_set.class_indices.keys()))
print("Accuracy score: ", accuracy_score(predIdxs, test_set.classes))

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
# show a nicely formatted classification report
test_set.classes
print(classification_report(test_set.classes, predIdxs,target_names=test_set.class_indices.keys()))

cm = confusion_matrix(test_set.classes, predIdxs)
total = sum(sum(cm))
acc = (cm[0, 0] + cm[1, 1]) / total
sensitivity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1])
print(cm)
print("acc: {:.4f}".format(acc))
print("sensitivity: {:.4f}".format(sensitivity))
print("specificity: {:.4f}".format(specificity))



import matplotlib.pyplot as plt
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, color='red', label='Training loss')
plt.plot(epochs, val_loss, color='green', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


acc = history.history['acc']
val_acc = history.history['val_acc']
val_precision_cat = history.history['precision']
val_precision_bin = history.history['val_precision_2']
plt.plot(epochs, acc, color='blue', label='Training acc')
plt.plot(epochs, val_acc, color='green', label='Validation acc')
plt.plot(epochs, val_precision_bin, color='red', label='Val precision bin')
plt.plot(epochs, val_precision_cat, color='purple', label='Val precision cat')
plt.title('Training and validation accuracy/precision')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
