from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow import keras

iteration = 300
defined_learning_rate = 0.01


def run(data_model):
    model = Sequential()
    model.add(Flatten(input_shape=(37,)))
    model.add(Dense(50, activation='relu'))  # LAYER 1
    model.add(Dense(5, activation='softmax'))  # OUTPUT TO 5
    model.summary()

    adamOpt = keras.optimizers.Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.999, amsgrad=False)
    model.compile(optimizer=adamOpt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(data_model.X_train, data_model.y_train, epochs=300, batch_size=10, validation_split=0.1, verbose=2)
