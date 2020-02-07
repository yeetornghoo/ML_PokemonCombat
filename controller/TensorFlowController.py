from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow import keras
import matplotlib
import matplotlib.pyplot as plt

iteration = 1000
defined_learning_rate = 0.01


def run(data_model):
    model = Sequential()
    model.add(Flatten(input_shape=(33,)))
    model.add(Dense(30, activation='relu'))
    model.add(Dense(18, activation='softmax'))
    model.summary()

    adamOpt = keras.optimizers.Adam(learning_rate=defined_learning_rate, beta_1=0.9, beta_2=0.999, amsgrad=False)
    model.compile(optimizer=adamOpt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(data_model.X_train, data_model.y_train, epochs=iteration, batch_size=10, validation_split=0.1, verbose=2)

    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    filename = "model_accuracy.png"
    plt.savefig(filename, dpi=50)
    plt.show()

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    filename = "model_loss.png"
    plt.savefig(filename, dpi=50)
    plt.show()