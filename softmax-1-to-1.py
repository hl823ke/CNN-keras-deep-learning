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


#Initialize the CNN

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
classifier.add(Dense(output_dim = 128, activation='relu'))
classifier.add(Dense(output_dim = 4, activation='softmax'))



#5 compile the CNN
classifier.compile(optimizer= 'Adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

#Image augmentation. Do this?

train_datagen = ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip=False)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('/Users/haikristianlethanh/Desktop/DP/Dataset/Train',target_size=(64,64),batch_size=32,class_mode='categorical')

test_set = test_datagen.flow_from_directory('/Users/haikristianlethanh/Desktop/DP/Dataset/Test', target_size= (64,64), batch_size=32, class_mode='categorical')

history = classifier.fit_generator(
    training_set,
    samples_per_epoch = 3661,
    nb_epoch= 15,
    validation_data= test_set,
    nb_val_samples= 915
)

test_set.reset()

predIdxs = classifier.predict_generator(test_set,steps=(915 // 32) + 1)
predIdxs = np.argmax(predIdxs, axis=1)

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
# show a nicely formatted classification report
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
plt.plot(epochs, acc, color='red', label='Training acc')
plt.plot(epochs, val_acc, color='green', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
