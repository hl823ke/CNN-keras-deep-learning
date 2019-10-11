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
import numpy as np

from keras_preprocessing.image import ImageDataGenerator
from keras.utils import plot_model, np_utils
import matplotlib.pyplot as plt

from sklearn.datasets import make_circles
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix


#Initialize the CNN
# define and fit the model
def get_model(trainX, trainy):
    classifier =  Sequential()
    #1 Convolution
    # Create many features map, 32 is common start point
    # input shape = set same format
    # we use rectifire activation function
    classifier.add(Convolution2D(32, 3, 3, input_shape=(64,64,3), activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(2,2)))
    classifier.add(BatchNormalization())
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
    classifier.add(Dense(output_dim = 128, input_dim=2, activation='relu'))
    classifier.add(Dense(output_dim = 1, activation='sigmoid'))
    #5 compile the CNN
    classifier.compile(optimizer= 'Adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    #classifier.fit_generator(training_set, samples_per_epoch = 3661, nb_epoch= 5, validation_data= test_set, nb_val_samples= 915)
    classifier.fit(trainX, trainy, epochs=300, verbose=0)
    return classifier





#Image augmentation. Do this?

train_datagen = ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip=False)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('/Users/haikristianlethanh/Desktop/DP/Dataset-1:N/1/Train',target_size=(64,64),batch_size=32,class_mode='binary')

test_set = test_datagen.flow_from_directory('/Users/haikristianlethanh/Desktop/DP/Dataset-1:N/1/Test', target_size= (64,64), batch_size=32, class_mode='binary')


history = get_model(training_set, test_set)


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
plt.plot(epochs, acc, color='red', label='Training acc')
plt.plot(epochs, val_acc, color='green', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# predict probabilities for test set
yhat_probs = history.predict(np.array(test_set), verbose=0)
# predict crisp classes for test set
yhat_classes = history.predict_classes(test_set, verbose=0)
# reduce to 1d array
yhat_probs = yhat_probs[:, 0]
yhat_classes = yhat_classes[:, 0]
 
# accuracy: (tp + tn) / (p + n)
accuracy = accuracy_score(testy, yhat_classes)
print('Accuracy: %f' % accuracy)
# precision tp / (tp + fp)
precision = precision_score(testy, yhat_classes)
print('Precision: %f' % precision)
# recall: tp / (tp + fn)
recall = recall_score(testy, yhat_classes)
print('Recall: %f' % recall)
# f1: 2 tp / (2 tp + fp + fn)
f1 = f1_score(testy, yhat_classes)
print('F1 score: %f' % f1)
 
# kappa
kappa = cohen_kappa_score(testy, yhat_classes)
print('Cohens kappa: %f' % kappa)
# ROC AUC
auc = roc_auc_score(testy, yhat_probs)
print('ROC AUC: %f' % auc)
# confusion matrix
matrix = confusion_matrix(testy, yhat_classes)
print(matrix)
