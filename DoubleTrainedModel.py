# =============================================================================
#  Nacitanie potrebnych kniznic
# =============================================================================
from keras.models import Sequential
from keras.layers import  Dropout, Conv2D, MaxPooling2D, Flatten, Dense
from keras_preprocessing.image import ImageDataGenerator
from keras.utils import plot_model
from keras.preprocessing.image import load_img, img_to_array
from keras.callbacks import ModelCheckpoint

from sklearn.metrics import classification_report, accuracy_score,confusion_matrix
from astropy.stats import sigma_clipped_stats
from skimage.color import rgb2gray
from skimage.transform import resize

import matplotlib.pyplot as plt
import keras_metrics as km
import numpy as np
import os
import split_folders
import shutil
import cv2
import time;

# =============================================================================
#  Definovane globalne premenne
# =============================================================================
seedNumbers = [2,3,5,7,11]
batchSize = 32
sigma = 3.0
activation_conv = 'relu'
presnost = []

# =============================================================================
#  Definovane cesty v globalnych premennych premenne
# =============================================================================
test_dir_name = 'cesta k testovacej datovej mnozine'
train_dir_name = 'cesta k trenovacej datovej mnozine'
path_to_dataset = 'cesta k datasetu'
splited_dir_path = 'cesta k adresaru, kde maju by ulozene rozdelene datove mnoziny'
path_to_unprocessed = 'cesta k adresaru, s neupravenymi datami. Stale davame jednu triedu napr.  /data/COMP/!!'
transformed_data_target ='cest k upravenemu datasetu napr. /data/COMP/'
path_to_model = 'cesta k ulozeniu natrenovaneho klasifikatora'
temp_fold = 'cesta k priecinku, kde sa ukladaju docasne subory. Po skonceni programu ich mozeme zmazat'


#=============================================================================
#   Pomocna funkcia sluziaca na odstránenie šumu pomocou hodnoty sigma.
#   Táto funkcia vracia danému dátovému bodu hodnotu 0, ak je jeho intenzita nižšia ako hranica inak vrati hodnotu spat
#   Vstup: hodnota pixlu a hranica
#   Vystup: nova hodnota
#=============================================================================
def clip(data,lim):
    data[data<lim] = 0.0
    return data 




#=============================================================================
#   Funkcia sluziaca na predspracovanie dat
#   Vstup: cesta k priecinku s neupravenymi datami a cesta k datasetu s upravenymi datami
#   Vystup: nova upravena datova mnozina
#=============================================================================
def preproces_img(path_to_unprocessed, dest_preprocessed):
    #   Inicializacia priecinku
    directory = os.fsencode(path_to_unprocessed)
    #   Prechadzanie snimok v danom priecinku
    for file in os.listdir(directory):
        #   Inicializacia filu
        filename = os.fsdecode(file)
        path_to_file = path_to_unprocessed + filename
        img = load_img(path_to_file)
        #   Prevod snimku na pole bodov
        img = img_to_array(img)
        #   Vypocet standardnej odchylky
        mean, median, std = sigma_clipped_stats(img, sigma=sigma, maxiters=10)
        #   Odstranenie sumu pomocou hodnoty sucinu sigma hodonty a standardnej odchylky
        img = clip(img,std*sigma)   
        #   Uprava rozmerov snimku
        if img.shape[0] !=150 or img.shape[1] !=150:
            img = resize(img, (150,150))
        # Prevod snimku na gray scale
        img = rgb2gray(img)  
        status = cv2.imwrite(dest_preprocessed, img)
        return status
        continue
     
#=============================================================================
#   Funkcia sluziaca na kontrolu ucenia
#   Vstup: klasifikator
#   Vystup: grafy celkovej uspesnosti a chyby ucenia
#=============================================================================
def trainingLoss(history):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('')
    plt.ylabel('celková úspešnosť')
    plt.xlabel('počet epoch')
    plt.legend(['trénovanie', 'validácia'], loc='upper left')
    plt.show()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('')
    plt.ylabel('chyba')
    plt.xlabel('počet epoch')
    plt.legend(['trénovanie', 'validácia'], loc='upper left')
    plt.show()

#=============================================================================
#   Funkcia sluziaca na vyhodnotenie metrik klasifikatora na testovacej datovej mnozine
#   Vstup: klasifikator
#   Vystup: grafy celkovej uspesnosti a chyby ucenia
#=============================================================================
def metrics(test_set, classifier):
    test_set.reset()
    numfiles = sum([len(files) for r, d, files in os.walk(test_dir_name)])
    print(numfiles)
    #   klasifikacia testovacej mnoziny
    Y_pred = classifier.predict_generator(test_set,steps=(numfiles // 32) + 1)
    print(Y_pred.shape)
    #   skutocne triedy distribucie testovacej mnoziny
    classes = test_set.classes[test_set.index_array]
    print(classes)
    test_set.classes
    y_pred = np.argmax(Y_pred, axis=-1)
    #   vypocet celkovej uspesnosti, presnosti klasifikacie
    accuracy = accuracy_score(y_pred, test_set.classes)
    print("Accuracy score: ", accuracy)
    y_pred
    #   vypocet chybovej matice
    cm = confusion_matrix(test_set.classes, y_pred)
    print(cm)
    print(classification_report(test_set.classes, y_pred,target_names=test_set.class_indices.keys()))
    return accuracy
    
#=============================================================================
#   Funkcia sluziaca na inicializaciu modelu konvolucnej siete
#   Vstup: 
#   Vystup: Klasifikator
#=============================================================================
def initializeModel():
    classifier =  Sequential()
    classifier.add(Conv2D(32, (3, 3), input_shape=(64,64,1), activation=activation_conv))
    classifier.add(MaxPooling2D(pool_size=(2,2)))
    classifier.add(Conv2D(64, (3, 3),input_shape=(64,64,1), activation=activation_conv))
    classifier.add(MaxPooling2D(pool_size=(2,2)))
    classifier.add(Conv2D(194, (3, 3),input_shape=(64,64,1), activation=activation_conv))
    classifier.add(MaxPooling2D(pool_size=(2,2)))
    classifier.add(Dropout(0.2))
    classifier.add(Flatten())
    classifier.add(Dense(units = 194, activation='relu'))
    classifier.add(Dense(units = 4, activation='softmax'))
    recall = km.binary_recall(label=0)
    precision = km.binary_precision(label=1)
    c_precision = km.categorical_precision()
    classifier.compile(optimizer= 'Adam', loss = 'categorical_crossentropy', metrics = [c_precision, 'accuracy', recall, precision, ])   
    classifier.summary()
    classifier.metrics_names
    return classifier


#=============================================================================
#   HLAVNA FUNKCIA
#=============================================================================
def main():   
    #=============================================================================
    #   Volanie funkcie predspracovania dat ak je potrebne data predspracovat
    #=============================================================================
    #  preproces_img(path_to_unprocessed, path_to_dataset)
    
    # Iteracia cez retazec seedov
    for i in seedNumbers:
        #   Nahodne rozdelenie datovej mnoziny podla seedu
        split_folders.ratio(path_to_dataset, output=splited_dir_path, seed=i, ratio=(.8, .2)) # default values
        #   Volanie funkcie, ktora nam vrati vytvoreny klasifikator
        classifier = initializeModel()
        #   Zobrazenie parametrov modelu
        plot_model(classifier, to_file='model.png', show_shapes=True, show_layer_names=True)
        numfiles = sum([len(files) for r, d, files in os.walk(path_to_dataset)])
        #   Inicializacia suborov na ukladanie modelov
        classifier.save(temp_fold + 'first_train.hdf5')
        classifier.save(temp_fold + 'second_train.hdf5')
        #   Nastavenie callbacku na hodnotu chyby. Tymto callbackom sledujeme vyvoj chybovej funkcie na trenovacej/validacnej mnozine, na zaklade ktorej ukladame model s jej minimalnou hodnotou
        checkpoint_callback1 = ModelCheckpoint(temp_fold + 'first_train.hdf5', monitor='loss', verbose=1, save_best_only=True, mode='min')
        checkpoint_callback2 = ModelCheckpoint(temp_fold + 'second_train.hdf5', monitor='loss', verbose=1, save_best_only=True, mode='min')
        #   Inicializacia augmentacie dat a naciatnie trenovacej datovej mnoziny
        train_datagen_1 = ImageDataGenerator(brightness_range=[1,1.5])
        train_datagen_2 = ImageDataGenerator(horizontal_flip=True,vertical_flip=True)
        training_set_1 = train_datagen_1.flow_from_directory(train_dir_name,target_size=(64,64), batch_size=batchSize,class_mode='categorical',subset='training',color_mode ='grayscale')
        training_set_2 = train_datagen_2.flow_from_directory(train_dir_name,target_size=(64,64),batch_size=batchSize,class_mode='categorical',color_mode ='grayscale')
        #   Nacitanie validacnej a testovacej datovej mnoziny
        test_datagen = ImageDataGenerator(validation_split=0.1)

        test_set = test_datagen.flow_from_directory(test_dir_name,
                                                    target_size= (64,64),
                                                    batch_size=batchSize,
                                                    class_mode='categorical',
                                                    color_mode ='grayscale',
                                                    shuffle=True)
        validation_set_1 = test_datagen.flow_from_directory(test_dir_name,target_size=(64,64), batch_size=batchSize,class_mode='categorical',subset='validation',color_mode ='grayscale')
    #=============================================================================
    #   TRENOVANIE MODELU
    #=============================================================================
        start = time.time()
        history = classifier.fit_generator(training_set_1,validation_data=validation_set_1, steps_per_epoch=numfiles/batchSize, nb_epoch=50, callbacks=[checkpoint_callback1])  
        classifier.load_weights(temp_fold + "first_train.hdf5")
        trainingLoss(history)
        metrics(test_set, classifier)
        history = classifier.fit_generator(training_set_2,validation_data=validation_set_1, steps_per_epoch=numfiles/batchSize, nb_epoch=50, callbacks=[checkpoint_callback2])   
        classifier.load_weights(temp_fold + "second_train.hdf5")
        end = time.time()
        hours, rem = divmod(end-start, 3600)
        minutes, seconds = divmod(rem, 60)
        print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))
    #=============================================================================
    #   TESTOVANIE MODELU
    #=============================================================================
        trainingLoss(history)
        start = time.time()
        acc_av= metrics(test_set, classifier)
        end = time.time()
        hours, rem = divmod(end-start, 3600)
        minutes, seconds = divmod(rem, 60)
        print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))
        classifier.save(path_to_model + str(acc_av) +'model.hdf5')
        presnost.append(metrics(test_set, classifier))
    # =============================================================================
    #     ## ODSTRANENIE NAHODNEHO ROZDELENIA
    # =============================================================================
        shutil.rmtree(splited_dir_path)
        print('Removed')
    

 
main()


