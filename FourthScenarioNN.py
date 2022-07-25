# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#https://machinelearningmastery.com/tensorflow-tutorial-deep-learning-with-tf-keras/

# mlp for binary classification
#ADJUSTMENTS MADE: import Dataframe to use the iloc function from pandas
from pandas import DataFrame
#END OF ADJUSTMENTS IN THIS PART
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from numpy import ravel
from numpy import min
from numpy import max
from pandas import set_option
import time
set_option("display.max_rows", None, "display.max_columns", None)
#END OF ADJUSTMENTS IN THIS PART


def load_data(file1,file2,percent):
    # load the dataset
    #X samples:
    X = read_csv(str(file1),header=None)
    X = X.iloc[:int(X.shape[0]*percent/100),:int(X.shape[1]*percent/100)]
    X = (X-min(X))/(max(X)-min(X))
    # ensure all data are floating point values
    X = X.astype('float32')
    #Y labels:
    y = read_csv(str(file2),header=None).to_numpy()
    y = y[:int(y.shape[0]*percent/100)]
    #ADJUSTMENTS MADE: conversion from dataFrame to numpy array and conversion to 1D array
    y = ravel(y)
    y = LabelEncoder().fit_transform(y)
    #END OF ADJUSTMENTS IN THIS PART
    return X, y


def splitting(X,y):
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.02, random_state=666)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, train_size=0.98, random_state=666)
    #END OF ADJUSTMENTS IN THIS PART
    # determine the number of input features
    n_features = X_train.shape[1]
    return X_train, y_train, X_val, y_val, X_test, y_test, n_features

def hidden_layers(n_features):
    model = Sequential()
    model.add(Dense(1000, activation='relu', kernel_initializer='he_normal', input_shape=(n_features,)))
    model.add(Dense(units=64,kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4), bias_regularizer=regularizers.l2(1e-4), activity_regularizer=regularizers.l2(1e-5)))
    model.add(BatchNormalization())
    model.add(Dropout(0.1))
    model.add(Dense(1, activation='sigmoid')) #activation='sigmoid'
    # summarize the model
    model.summary()
    return model


def plotting(model):
    # summarize the model
    plot_model(model, 'model.png', show_shapes=True)
    #loss = The loss function should be minimized to guide the model to the correct direction
    #optimizer = how the model is updated according the saw data and the loss function
    #metrics = monitors the training and test steps

def propagate(model, X_train, y_train, X_val, y_val, X_test, y_test):
    # compile the model
    sgd = SGD(learning_rate= 0.001, momentum=0.8)
    model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy'])
    # configure early stopping
    es = EarlyStopping(monitor='val_loss', patience=5)
    # fit the model = neural network training start
    our_zero1 = time.time()
    history = model.fit(X_train, y_train, epochs=150, batch_size=5, verbose=1, callbacks=None, validation_split=None, validation_data=(X_val,y_val), shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0)
    get_secs1 = time.time() - our_zero1
    print("Training execution time: ",get_secs1)
    # evaluate the model
    our_zero2 = time.time()
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    get_secs2 = time.time() - our_zero2
    print("Test execution time: ",get_secs2)

    print('Test Loss: %.3f' % loss)
    print('Test Accuracy: %.3f' % acc)
    return history.history['accuracy'], history.history['val_accuracy'], history.history['loss'], history.history['val_loss'], loss, acc


def plot(loss, acc):

    fig, (ax1, ax2) = plt.subplots(2)

    # plot learning curves
    ax1.set_title('Test Loss Average In Each Percentual Amount of Data')
    ax1.set(xlabel = 'Dataset Amount (%)', ylabel = 'Avg')
    ax1.plot(loss, color='green')
    ax1.label_outer()

    ax2.set_title('Test Accuracy Average In Each Percentual Amount of Data')
    ax2.set(xlabel = 'Dataset Amount (%)', ylabel = 'Avg')
    ax2.plot(acc, color='blue')
    ax2.label_outer()

    plt.savefig('teste4.jpg',format='jpg', dpi=1600)
    plt.savefig('teste4.jpg',format='jpg', dpi=1600)

def start(file1,file2):
    amount = 10
    arr_loss = [None]*amount
    arr_acc = [None]*amount
    for percent in range(10,101,10):
        X,y = load_data(file1,file2,percent)
    f1 = open("sampleFFT.csv","w")
    f1.write(str(X))
    f1.close()
    

def main():
    start('datasetEEG_fft.csv','datasetEEG_label.csv')
    start('datasetEEG.csv','datasetEEG_label.csv')

if __name__ == "__main__":
	main()
