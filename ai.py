from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt


def createModel():
    model = Sequential()
    model.add(Dense(10, input_shape=(9,), activation="relu"))
    model.add(Dense(10, activation="relu"))
    model.add(Dense(10, activation="relu"))
    model.add(Dense(9, activation="softmax"))

    model.compile(loss=categorical_crossentropy,
                  optimizer=Adam(),
                  metrics=['accuracy'])
    print(model.summary())

    return model


def trainModel(model, epochs):
    x_train = np.loadtxt("xvalues.txt")
    print(x_train.shape)
    y_train = to_categorical(np.loadtxt("yvalues.txt"))

    modelHistory = model.fit(x_train, y_train, epochs=epochs, verbose=0, validation_split=0.2, shuffle=True)

    return model, modelHistory


def makePrediction(model, x):
    return model.predict(x)


def plotTraining(modelHistory):
    # summarize history for accuracy
    plt.plot(modelHistory.history['acc'])
    plt.plot(modelHistory.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    # summarize history for loss
    plt.plot(modelHistory.history['loss'])
    plt.plot(modelHistory.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
