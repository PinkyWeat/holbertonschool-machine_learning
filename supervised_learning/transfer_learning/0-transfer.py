#!/usr/bin/env python3
""" Transfer Learning """

import tensorflow.keras as K


def preprocess_data(X, Y):
    """ script trains CNN to classify CIFAR 10 dataset """
    X_p = K.applications.resnet50.preprocess_input(X)
    Y_p = K.utils.to_categorical(Y, num_classes=10)
    return X_p, Y_p


# load & preprocess the data
(x_train, y_train), (x_test, y_test) = K.datasets.cifar10.load_data()
x_train, y_train = preprocess_data(x_train, y_train)
x_test, y_test = preprocess_data(x_test, y_test)

# Define model
resize_layer = K.layers.Lambda(lambda image: K.backend.resize_images(image, 7, 7, data_format='channels_last', interpolation='bilinear'))
base_model = K.applications.ResNet50(include_top=False, weights='imagenet')
base_model.trainable = False

model = K.Sequential([
    resize_layer,
    base_model,
    K.layers.Flatten(),
    K.layers.BatchNormalization(),
    K.layers.Dense(256, activation='relu', kernel_initializer=K.initializers.HeNormal()),
    K.layers.Dropout(0.3),
    K.layers.BatchNormalization(),
    K.layers.Dense(128, activation='relu', kernel_initializer=K.initializers.HeNormal()),
    K.layers.Dropout(0.3),
    K.layers.BatchNormalization(),
    K.layers.Dense(64, activation='relu', kernel_initializer=K.initializers.HeNormal()),
    K.layers.Dropout(0.3),
    K.layers.Dense(10, activation='softmax', kernel_initializer=K.initializers.HeNormal())
])

model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=['accuracy'])

best_model_save = K.callbacks.ModelCheckpoint('cifar10.h5', monitor='val_accuracy', verbose=1, save_best_only=True)

model.fit(x_train, y_train, epochs=3, batch_size=32, validation_data=(x_test, y_test), callbacks=[best_model_save])
model.save('cifar10.h5')