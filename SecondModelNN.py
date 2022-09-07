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
from numpy import ravel
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import SGD
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def file_input(file_X, file_y, modelname):
	# load the dataset
	# input and output columns
	X = read_csv(file_X,header=None)

	print(X.to_numpy().min())
	print(X.to_numpy().max())

	#Normalization Min-Max
	X = (X - X.min(axis=0))/(X.max(axis=0) - X.min(axis=0))

	#Xis = read_csv('datasetEEG.csv',header=None)
	#X = Xis.values[:,1:2000]
	print(X.to_numpy().min())
	print(X.to_numpy().max())

	y = read_csv(file_y,header=None)
	y = y.to_numpy()
	y = ravel(y)
	# ensure all data are floating point values
	X = X.astype('float32')
	# encode strings to integer
	y = LabelEncoder().fit_transform(y)
	# split into train and test datasets
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
	#print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
	# determine the number of input features
	n_features = X_train.shape[1]

	print(X.to_numpy().min())
	print(X.to_numpy().max())

	# define model
	model = Sequential()
	model.add(Dense(100, activation='relu', kernel_initializer='he_normal', input_shape=(n_features,)))
	model.add(BatchNormalization())
	model.add(Dropout(0.5))
	model.add(Dense(1, activation='sigmoid'))

	# summarize the model
	model.summary()
	# summarize the model
	plot_model(model, modelname, show_shapes=True)

	# compile the model
	sgd = SGD(learning_rate=0.001, momentum=0.8)
	model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy'])
	#model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

	# configure early stopping
	es = EarlyStopping(monitor='val_loss', patience=5)

	# fit the model
	history = model.fit(X_train, y_train, epochs=150, batch_size=32, verbose=0, validation_split=0.3)
	# evaluate the model
	loss, acc = model.evaluate(X_test, y_test, verbose=0)
	print('Test Accuracy: %.3f' % acc)
	print('Test Loss : %.3f' % loss)

	return history.history['accuracy'], history.history['val_accuracy'], history.history['loss'], history.history['val_loss']


def plot(acc1, val_acc1, loss1, val_loss1, acc2, val_acc2, loss2, val_loss2):

	fig, axs = plt.subplots(2,2)

	# plot learning curves
	axs[0,0].set_title('A')
	axs[0,0].set(xlabel = 'Época', ylabel = 'Acurácia')
	axs[0,0].plot(acc1, label='accuracy', color='#006400')
	axs[0,0].plot(val_acc1, label='val_accuracy', color='#66CDAA')
	axs[0,0].label_outer()

    
    # plot learning curves
	axs[0,1].set_title('B')
	axs[0,1].set(xlabel = 'Época', ylabel = 'Acurácia')
	axs[0,1].plot(acc2, label='accuracy', color='#006400')
	axs[0,1].plot(val_acc2, label='val_accuracy', color='#66CDAA')
	axs[0,1].label_outer()


	axs[1,0].set_title('C')
	axs[1,0].set(xlabel = 'Época', ylabel = 'Entropia Cruzada')
	axs[1,0].plot(loss1, label='loss', color='red')
	axs[1,0].plot(val_loss1, label='val_loss', color='magenta')
	axs[1,0].label_outer()


	axs[1,1].set_title('D')
	axs[1,1].set(xlabel = 'Época', ylabel = 'Entropia Cruzada')
	axs[1,1].plot(loss2, label='loss', color='red')
	axs[1,1].plot(val_loss2, label='val_loss', color='magenta')
	axs[1,1].label_outer()

	dark_green_patch = mpatches.Patch(color='#006400', label='acur')
	aquamarine_patch = mpatches.Patch(color='#66CDAA', label='val_acur')
	red_patch = mpatches.Patch(color='red', label='perda')
	magenta_patch = mpatches.Patch(color='magenta', label='val_perda')
	plt.legend(handles=[dark_green_patch, aquamarine_patch, red_patch, magenta_patch])
	plt.show()
	fig.savefig('teste2_pt.jpg',format='jpg', dpi=1600)
    


def main():

    file_input('datasetEEG.csv', 'datasetEEG_label.csv', 'model_norm.png')
    file_input('datasetEEG_fft.csv', 'datasetEEG_label.csv', 'model_fft_norm.png')


if __name__ == "__main__":
    main()
