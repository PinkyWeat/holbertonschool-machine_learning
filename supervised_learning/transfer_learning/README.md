# Transfer Learning

WIP

---

(venv) <myUser> transfer_learning % python3 0-main.py  

Downloading data from https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz

170498071/170498071 [==============================] - 19s 0us/step

Downloading data from https://storage.googleapis.com/tensorflow/keras-applications/resnet/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5

94765736/94765736 [==============================] - 8s 0us/step

Epoch 1/3

1563/1563 [==============================] - ETA: 0s - loss: 0.6231 - accuracy: 0.7978     

Epoch 1: val_accuracy improved from -inf to 0.90140, saving model to cifar10.h5

<filePath>training.py:3000: UserWarning: You are saving your model as an HDF5 file via `model.save()`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')`.

1563/1563 [==============================] - 2210s 1s/step - loss: 0.6231 - accuracy: 0.7978 - val_loss: 0.2951 - val_accuracy: 0.9014

Epoch 2/3

1563/1563 [==============================] - ETA: 0s - loss: 0.3263 - accuracy: 0.8958     

1563/1563 [==============================] - 2313s 1s/step - loss: 0.3263 - accuracy: 0.8958 - val_loss: 0.2806 - val_accuracy: 0.9101

Epoch 3/3

1563/1563 [==============================] - ETA: 0s - loss: 0.2341 - accuracy: 0

1563/1563 [==============================] - 3318s 2s/step - loss: 0.2341 - accuracy: 0.9237 - val_loss: 0.2991 - val_accuracy: 0.9042

79/79 [==============================] - 326s 4s/step - loss: 0.2991 - accuracy: 0.9042
