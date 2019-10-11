import numpy as np
import matplotlib.pyplot as plt
np.random.seed(12345)

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from keras.datasets import boston_housing

(x_train, y_train), (x_test, y_test) = boston_housing.load_data()
print(x_train.shape, x_test.shape)

#Ide o jednoduchu doprednu neuronovu sier
# 13 neuronov pricom kazdy zodpoveda 1 vstupnemu parametru
model = Sequential()
# Vytvorime 10 vstupnych neuronov
# Podla poctu atributov
# Urobi sa full-conection
# Aktivacna funkcia je tanh: tangens hyperbolicky
# predikujeme 1 hodnotu takze na vystupnej vrstve 1 neuron
model.add(Dense(10, input_dim=13, activation='tanh'))
model.add(Dense(1, activation='linear'))

#dolezity je pocet parametrov, kedze ide o fullconection  tak mame 10*13+ bias kt je rovny 10
model.summary()

#Nastavujeme rychlost ucenia
#Nastavujeme chybovu funkciu
# Pri klasifikacii je najlepsia kriva entropia inak v tomto pripade je toto lepsie
sgd = SGD(0.01)
model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['mean_absolute_error', "accuracy"])

#
f = model.fit(x_train, y_train, epochs=25, batch_size=10)

plt.figure(1)
plt.subplot(211)
plt.plot(f.history['loss'])
plt.subplot(212)
plt.plot(f.history['mean_absolute_error'])

scores = model.evaluate(x_test, y_test)
print('MSE: {:.4f}, MAE: {:.4f}'.format(scores[0], scores[1]))