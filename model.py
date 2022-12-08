import tensorflow as tf

class MyModel():
    
    def __init__(self, input_shape):
        self.input_shape=input_shape

    def build_model(self,vgg_model=False):
        if(vgg_model):
            model = tf.keras.applications.vgg16.VGG16(include_top=False, input_shape=self.input_shape)
            for layer in model.layers:
                layer.trainable=False
            flat1 = tf.keras.layers.Flatten()(model.layers[-1].output)
            class1 = tf.keras.layers.Dense(128, activation='relu', kernel_initializer='he_uniform')(flat1)
            output = tf.keras.layers.Dense(1, activation='sigmoid')(class1)
            model = tf.keras.models.Model(inputs=model.inputs, outputs=output)

        else:
            model = tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=self.input_shape),
            tf.keras.layers.Conv2D(32, (3,3)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation("relu"),
            tf.keras.layers.MaxPooling2D((2,2)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Conv2D(64, (3,3)), 
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation("relu"),
            tf.keras.layers.MaxPooling2D((2,2)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Conv2D(128, (3,3)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation("relu"),
            tf.keras.layers.MaxPooling2D((2,2)), 
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')])        
        model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['acc'])
        return model

        
        