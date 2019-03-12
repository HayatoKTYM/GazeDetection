from tensorflow import keras
import tensorflow as tf

def CNN():
    input = keras.layers.Input(shape=(32,96,1))
    conv1 = keras.layers.Conv2D(32, 3,activation='relu',name='conv1')(input)
    pool1 = keras.layers.MaxPooling2D((2, 2),name='pool1')(conv1)
    conv2 = keras.layers.Conv2D(32, 3,activation='relu',name='conv2')(pool1)
    pool2 = keras.layers.MaxPooling2D((2, 2),name='pool2')(conv2)
    bn = keras.layers.BatchNormalization(name='bn1')(pool2)
    x = keras.layers.Flatten()(bn)
    x = keras.layers.Dense(256,activation='relu',name='dense1')(x)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Dense(128,activation='relu',name='dense2')(x)
    x = keras.layers.Dropout(0.5)(x)
    output = keras.layers.Dense(1,activation='sigmoid')(x)

    gazemodel = keras.models.Model(inputs=input,outputs=output)
    optimizer = tf.train.AdamOptimizer(1e-4)
    gazemodel.compile(loss='binary_crossentropy',
                        optimizer=optimizer,
                        metrics=['accuracy'])
    gazemodel.summary()
    return gazemodel

if __name__ == "__main__":
    model = CNN()
