#! /usr/bin/python3

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras import backend

import numpy as np
import matplotlib.pyplot as plt

from load_data import load_data

def run_cnn(classes=3, iters=6):
    num_classes = classes
    epochs      = iters

    x_train, x_test, y_train, y_test = load_data('dataset_resize')
    [train_simples, img_rows, img_cols] = np.shape(x_train)

    # Image data format
    if backend.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    # Data to 0-1
    x_train  = x_train.astype('float32')
    x_test   = x_test.astype('float32')
    x_train /= 255
    x_test  /= 255

    # Print info of data
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    # Set up cnn model
    model = Sequential()
    # Layer1 -- C1
    model.add(Conv2D(16, kernel_size=(5, 5),
                     activation='relu',
                     input_shape=input_shape))
    # Layer2 -- P1
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Layer3 -- C2
    model.add(Conv2D(32, (3, 3), 
                     activation='relu'))
    # Layer4 -- P2
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dropout(0.25))
    model.add(Flatten()) 
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    # Full Connect & Classify
    model.add(Dense(num_classes, activation='softmax'))

    # Compile CNN
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    # Training 
    history = model.fit(x_train, y_train,
                        epochs=epochs,
                        verbose=1,
                        validation_data=(x_test, y_test))

    # Evalute the model by test data
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    #----------------------------------------------
    # Plot The training process
    #----------------------------------------------
    colors   = ['r','g','b','k','c','m','y']
    markers  = ['o','s','p','*','+','x','D','d','v','^','<','>']

    # Accuracy
    figure  = plt.figure("accuracy")
    fig_acc = figure.add_subplot(111)
    fig_acc.plot(history.history['accuracy'], marker=markers[0], c=colors[0])
    fig_acc.plot(history.history['val_accuracy'],  marker=markers[1], c=colors[1])
    plt.title('Model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')

    # Loss
    e_fig  = plt.figure("Loss Curve")
    e_plot = e_fig.add_subplot(111)
    e_plot.plot(history.history['loss'], marker=markers[0], c=colors[0])
    e_plot.plot(history.history['val_loss'], marker=markers[1], c=colors[1])
    plt.title('Model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')
    
    plt.show()

if __name__=='__main__':
    run_cnn()
