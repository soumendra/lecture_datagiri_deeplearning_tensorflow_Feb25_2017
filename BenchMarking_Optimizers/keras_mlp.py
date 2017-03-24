""" A Keras Convolution function """

""" This will be a simple wrapper around the Keras module

scope:
1- Give the entire data and labels as numpy array
2. model = The entire model should be defined
3. compile = the entire keras model should be compiled
4. train = the entire training should be completed
5. inference = Given new data it should predict the output
"""

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop

class Mnist:
    def __init__(self, data, target, test_data, test_target):
        self.data = data
        self.target = target
        self.test_data = test_data
        self.test_target = test_target
        self.model = None
        self.inference = None

    def build_graph(self):
        model = Sequential()
        model.add(Dense(256,activation = "relu", input_shape=(784,)))
        model.add(Dropout(0.2))
        model.add(Dense(128, activation = "relu"))
        model.add(Dropout(0.2))
        model.add(Dense(10, activation = "softmax"))
        self.model = model

    def compile_graph(self):
        return self.model.compile(loss="categorical_crossentropy",
        optimizer = RMSprop(),
        metrics = ["accuracy"])

    def train_model(self, batch_size=32, epoch=20):
        self.model.fit(self.data, self.target, epochs = epoch, batch_size = batch_size,verbose =1 , validation_data=(self.test_data, self.test_target))
        return print("[Model training completed]")

    def evaluate_model(self,  input_data):
        predict = self.model.predict(input_data, batch_size=1, verbose =0)
        return predict
