# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 10:05:08 2018
Modified on Tue Nov 12 21:19:10 2019
@author1: Yongrui Huang
@author2: Ruixin Lee
"""

import keras.layers as L
import keras
import matplotlib.pyplot as plt


# get_model()
def get_model():
    print("build_model.py..get_model().start...")

    '''
    return:
        base model for training
        
    #--------------------------#
        Does this function prepare for transfer learning?      From Ruixin Lee
    #--------------------------#
    
    '''

    input = L.Input(shape=(48, 48, 1))

    x = L.Conv2D(32, (3, 3), activation='relu', padding='same')(input)
    x = L.Conv2D(32, (3, 3), activation='relu')(x)
    x = L.Conv2D(64, (3, 3), activation='relu')(x)
    x = L.Dropout(0.5)(x)
    
    x = L.MaxPooling2D(pool_size=(3, 3))(x)

    # Flatten vector
    x = L.Flatten(name='bottleneck')(x)
    x = L.Dense(64, activation='relu')(x)
    x = L.Dropout(0.5)(x)
    output = L.Dense(7, activation='softmax')(x)
  
    model = keras.Model(input=input, output = output)
    print("model1__summary:")
    print(model.summary())
    # model.compile(optimizer=keras.optimizers.Adadelta(), loss='categorical_crossentropy', metrics=['accuracy'])

    print("build_model.py..get_model().end...")
    return model
# get_model()

def get_model2():
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, Activation, Flatten
    from keras.layers import Conv2D, MaxPool2D, ZeroPadding2D

    import keras

    model = Sequential()
    model.add(
        keras.layers.Input(
            shape=(48, 48, 1)
        )
    )
    model.add(
        Conv2D(
            filters=32,
            kernel_size=(3, 3),
            activation='relu',
            padding="same"
        )
    )
    model.add(
        Conv2D(
            filters=32,
            kernel_size=(3, 3),
            activation='relu'
        )
    )
    model.add(
        Conv2D(
            filters=64,
            kernel_size=(3, 3),
            activation='relu'
        )
    )
    model.add(Dropout(0.5))
    model.add(MaxPool2D(pool_size=(3, 3)))
    model.add(Flatten(name="bottleneck"))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(rate=0.5))
    model.add(Dense(7, activation='softmax'))
    print("model2__summary:")
    print(model.summary())
    # model.compile(
    #     loss='categorical_crossentropy',
    #     optimizer=keras.optimizers.Adadelta(), metrics=['accuracy']
    # )
    #-------------------------#
    # input = L.Input(shape=(48, 48, 1))
    #
    # x = L.Conv2D(32, (3, 3), activation='relu', padding='same')(input)
    # x = L.Conv2D(32, (3, 3), activation='relu')(x)
    # x = L.Conv2D(64, (3, 3), activation='relu')(x)
    # x = L.Dropout(0.5)(x)
    #
    # x = L.MaxPooling2D(pool_size=(3, 3))(x)
    #
    # # Flatten vector
    # x = L.Flatten(name='bottleneck')(x)
    # x = L.Dense(64, activation='relu')(x)
    # x = L.Dropout(0.5)(x)
    # output = L.Dense(7, activation='softmax')(x)
    #
    # model = keras.Model(input=input, output=output)
    #
    # model.compile(optimizer=keras.optimizers.Adadelta(), loss='categorical_crossentropy', metrics=['accuracy'])
    #
    # print("build_model.py..get_model().end...")
    return model


# plot_training(history, filename)
def plot_training(history, filename):
    print("build_model.py..plot_training(history, filename).start...")

    '''
        plot the train data image
    '''
    
    output_acc = history.history['acc']
    val_output_acc = history.history['val_acc']

    output_loss = history.history['loss']
    val_output_loss = history.history['val_loss']
    
    epochs = range(len(val_output_acc))
    
    plt.figure()
    plt.plot(epochs, output_acc, 'b-', label='train accuracy')
    plt.plot(epochs, val_output_acc, 'r-', label='validation accuracy')
    plt.legend(loc='best')
    plt.title('Training and validation accuracy')
    plt.savefig(filename+'_accuray'+'.png')
    
    plt.figure()
    plt.plot(epochs, output_loss, 'b-', label='train loss')
    plt.plot(epochs,  val_output_loss, 'r-', label='validation loss')
    plt.legend(loc='best')
    plt.title('Training and validation loss')
    plt.savefig(filename+'_loss' + '.png')

    print("build_model.py..plot_training(history, filename).end...")
    return 0
# plot_training(history, filename)


# __main__
if __name__ == '__main__':
    print("build_model.py..__main__.start...")
    get_model()
    get_model2()
    print("build_model.py..__main__.end...")
# __main__
