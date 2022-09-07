# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#https://machinelearningmastery.com/tensorflow-tutorial-deep-learning-with-tf-keras/

# mlp for binary classification
from pandas import read_csv
from pandas import set_option
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import SGD
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tensorflow.keras import regularizers
from matplotlib import pyplot
#ADJUSTMENTS MADE: ravel importing from numpy library
from numpy import ravel
from numpy import min
from numpy import max
from pandas import set_option
set_option("display.max_rows", None, "display.max_columns", None)
#END OF ADJUSTMENTS IN THIS PART
import time


SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 15

plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.rc('legend', fontsize=8)


def file_input(file_X, file_y, modelname):
	# load the dataset
	# input and output columns
	X = read_csv(file_X,header=None)
	norm_X = (X-min(X))/(max(X)-min(X))
	y = read_csv(file_y,header=None)
	#ADJUSTMENTS MADE: conversion from dataFrame to numpy array and conversion to 1D array
	yline = y.to_numpy()
	one_dim_y = ravel(yline)
	#END OF ADJUSTMENTS IN THIS PART

	# ensure all data are floating point values
	norm_X = norm_X.astype('float32')
	one_dim_y = LabelEncoder().fit_transform(one_dim_y)
	# split into train and test datasets
	X_train_val, X_test, y_train_val, y_test = train_test_split(norm_X, one_dim_y, test_size=0.02, random_state=666)
	X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, train_size=0.98, random_state=666)
	#END OF ADJUSTMENTS IN THIS PART
	# determine the number of input features

	n_features = X_train.shape[1]

	# define model

	model = Sequential()
	model.add(Dense(1000, activation='relu', kernel_initializer='he_normal', input_shape=(n_features,)))
	model.add(Dense(units=64,kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4), bias_regularizer=regularizers.l2(1e-4), activity_regularizer=regularizers.l2(1e-5)))
	model.add(BatchNormalization())
	model.add(Dense(1, activation='sigmoid'))

	# summarize the model
	model.summary()
	# summarize the model
	plot_model(model, modelname, show_shapes=True)
	#loss = The loss function should be minimized to guide the model to the correct direction
	#optimizer = how the model is updated according the saw data and the loss function
	#metrics = monitors the training and test steps

	# compile the model
	sgd = SGD(learning_rate= 0.001, momentum=0.8)
	model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy'])
	#model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

	# configure early stopping
	es = EarlyStopping(monitor='val_loss', patience=5)
	# fit the model = neural network training start
	our_zero = time.time()
	history = model.fit(X_train, y_train, epochs=2, batch_size=5, verbose=1, callbacks=None, validation_split=None, validation_data=(X_val,y_val), shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0)
	execution_time = time.time() - our_zero
	# evaluate the model
	loss, acc = model.evaluate(X_test, y_test, verbose=0)
	#output.write(str(acc))
	print("Execution time: ", execution_time)
	print('Test Accuracy: %.3f' % acc)
	print('Test Loss : %.3f' % loss)

	return history.history['accuracy'], history.history['val_accuracy'], history.history['loss'], history.history['val_loss']


def plot(acc1, val_acc1, loss1, val_loss1, acc2, val_acc2, loss2, val_loss2):

	fig, axs = plt.subplots(2,2)

	# plot learning curves
	axs[0,0].set_title('A')
	axs[0,0].set(xlabel = 'Epoch', ylabel = 'Accuracy')
	axs[0,0].plot(acc1, label='accuracy', color='#006400')
	axs[0,0].plot(val_acc1, label='val_accuracy', color='#66CDAA')
	axs[0,0].label_outer()

    
    # plot learning curves
	axs[0,1].set_title('B')
	axs[0,1].set(xlabel = 'Epoch', ylabel = 'Accuracy')
	axs[0,1].plot(acc2, label='accuracy', color='#006400')
	axs[0,1].plot(val_acc2, label='val_accuracy', color='#66CDAA')
	axs[0,1].label_outer()


	axs[1,0].set_title('C')
	axs[1,0].set(xlabel = 'Epoch', ylabel = 'Cross Entropy')
	axs[1,0].plot(loss1, label='loss', color='red')
	axs[1,0].plot(val_loss1, label='val_loss', color='magenta')
	axs[1,0].label_outer()


	axs[1,1].set_title('D')
	axs[1,1].set(xlabel = 'Epoch', ylabel = 'Cross Entropy')
	axs[1,1].plot(loss2, label='loss', color='red')
	axs[1,1].plot(val_loss2, label='val_loss', color='magenta')
	axs[1,1].label_outer()

	dark_green_patch = mpatches.Patch(color='#006400', label='acc')
	aquamarine_patch = mpatches.Patch(color='#66CDAA', label='val_acc')
	red_patch = mpatches.Patch(color='red', label='loss')
	magenta_patch = mpatches.Patch(color='magenta', label='val_loss')
	plt.legend(handles=[dark_green_patch, aquamarine_patch, red_patch, magenta_patch])
	plt.show()
	fig.savefig('teste3.jpg',format='jpg', dpi=1600)
    


def main():

    acc1, val_acc1, loss1, val_loss1 = file_input('datasetEEG_extended.csv', 'datasetEEG_label_extended.csv', 'model_960202_norm.png')
    acc2, val_acc2, loss2, val_loss2 = file_input('datasetEEG_fft_extended.csv', 'datasetEEG_label_extended.csv', 'model_fft_960202_norm.png')

    plot(acc1, val_acc1, loss1, val_loss1, acc2, val_acc2, loss2, val_loss2)


if __name__ == "__main__":
	main()
