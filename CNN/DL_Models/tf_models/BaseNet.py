import tensorflow as tf
import tensorflow.keras as keras
from keras.callbacks import CSVLogger
import numpy as np
import logging
from config import config 

class prediction_history(tf.keras.callbacks.Callback):
    """
    Prediction history for model ensembles=
    """
    def __init__(self, validation_data):
        self.validation_data = validation_data
        self.predhis = []
        self.targets = validation_data[1]

    def on_epoch_end(self, epoch, logs={}):
        y_pred = self.model.predict(self.validation_data[0])
        self.predhis.append(y_pred)


class BaseNet:
    def __init__(self, loss, input_shape, output_shape, epochs=50, verbose=True, model_number=0):
        self.epochs = epochs
        self.verbose = verbose
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.loss = loss 
        self.model = self._build_model()

        # Compile the model depending on the task 
        if self.loss == 'bce':
            self.model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(learning_rate=config['learning_rate']), metrics=['accuracy'])
        else:
            raise ValueError("Choose valid loss for your task")

    # abstract method
    def _split_model(self):
        pass

    # abstract method
    def _build_model(self):
        pass

    def get_model(self):
        return self.model

    def save(self, path):
        self.model.save(path)

    def fit(self, X_train, y_train, X_val, y_val):
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)
        hist = self.model.fit(X_train, y_train, verbose=2, batch_size=self.batch_size, validation_data=(X_val, y_val),
                                  epochs=self.epochs, callbacks=[early_stop])

    def predict(self, testX):
        return self.model.predict(testX)
