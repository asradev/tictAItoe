import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Dense
from keras.losses import categorical_crossentropy
from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils import to_categorical


def create_model():
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


def train_model(model, epochs):
    x_train = np.loadtxt("xvalues.txt")
    print(x_train.shape)
    y_train = to_categorical(np.loadtxt("yvalues.txt"))

    model_history = model.fit(x_train, y_train, epochs=epochs, verbose=0, validation_split=0.2, shuffle=True)

    return model, model_history


def make_prediction(model, x):
    return model.predict(x)


def plot_training(model_history):
    # summarize history for accuracy
    plt.plot(model_history.history['acc'])
    plt.plot(model_history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    # summarize history for loss
    plt.plot(model_history.history['loss'])
    plt.plot(model_history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
