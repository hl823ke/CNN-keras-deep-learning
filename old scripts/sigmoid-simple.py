#initioalize network
import classifier as classifier
from keras.models import Sequential
# to add convuluion layer
from keras.layers import  Convolution2D
# proceed pooling layer
from keras.layers import MaxPooling2D
#pooling to large vector
from keras.layers import  Flatten
#used to add to full conected network
from keras.layers import Dense

from keras_preprocessing.image import ImageDataGenerator
from keras.utils import plot_model

#Initialize the CNN

classifier =  Sequential()

#1 Convolution
# Create many features map, 32 is common start point
# input shape = set same format
# we use rectifire activation function
classifier.add(Convolution2D(32, 3, 3, input_shape=(64,64,3), activation='relu'))

#2 Pooling
# It gets maximum value from features map and create pooled feature map, it call max pooling
# improve complexity and performance
# 2x2 we still keep it precise and dont loose ani information
classifier.add(MaxPooling2D(pool_size = (2,2)))

#3 Flattening
#Create vector
classifier.add(Flatten())

#4 Full connection
#hidden layer
# 128 needed experiments to decide correct number
classifier.add(Dense(output_dim = 128, activation='relu'))
classifier.add(Dense(output_dim = 1, activation='sigmoid'))

#5 compile the CNN
classifier.compile(optimizer= 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#Image augmentation. Do this?

train_datagen = ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip=False)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('/Users/haikristianlethanh/Desktop/DP/Dataset/Train',target_size=(64,64),batch_size=32,class_mode='binary')

test_set = test_datagen.flow_from_directory('/Users/haikristianlethanh/Desktop/DP/Dataset/Test',
                                            target_size= (64,64),
                                            batch_size=32,
                                            class_mode='binary')
classifier.fit_generator(
    training_set,
    samples_per_epoch = 1000,
    nb_epoch= 5,
    validation_data= test_set,
    nb_val_samples= 200
)
plot_model(classifier, to_file='model.png')
