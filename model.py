from tensorflow.python.keras.models import *
from tensorflow.python.keras.layers import *

class MyModel():
    
    def __init__(self, input_shape):
        self.height, self.width, self.channels = input_shape
        self.cnn_1=Conv2D(filters=128,kernel_size=3,padding="same",activation="relu")
        self.cnn_2=Conv2D(filters=128,kernel_size=3,padding="same",activation="relu")
        self.cnn_3=Conv2D(filters=64,kernel_size=3,padding="same",activation="relu")
        self.fcl = Flatten()
        self.dense_1 = Dense(units=64,activation="relu")
        self.dense_2 = Dense(units=16, activation='relu')
        self.dense_3 = Dense(units=1, activation='sigmoid')
        self.pool = MaxPooling2D(pool_size = 2)
        self.dropout = Dropout(0.5)

    def build_model(self):
        input = Input(shape=(self.height,self.width, self.channels))
        cnn_layer_1 = self.cnn_1(input)
        cnn_layer_2 = self.cnn_2(cnn_layer_1)
        pool_layer = self.pool(cnn_layer_2)
        cnn_layer_3 = self.cnn_3(pool_layer)
        fcl_start = self.fcl(cnn_layer_3)
        fcl_1 = self.dense_1(fcl_start)
        drop_layer = self.dropout(fcl_1)
        fcl_2 = self.dense_2(drop_layer)
        fcl_3 = self.dense_3(fcl_2)
        
        model = Model(inputs = input, outputs = fcl_3)
        
        return model