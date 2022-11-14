import tensorflow as tf

class MyModel():
    
    def __init__(self, input_shape):
        self.height, self.width, self.channels = input_shape

    def build_model(self):
        model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(self.height,self.width, self.channels)),
        tf.keras.layers.Conv2D(128, (3,3)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation("relu"),
        tf.keras.layers.Conv2D(64, (3,3)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation("relu"),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')])
        return model