#!/usr/bin/env python3
""" Forecast BTC """
import tensorflow as tf
from tensorflow import keras
from preprocess_data import preprocess_data


def implement_dataset(df, sequence_length):
    """ BTC WIP """
    data = df.values
    targets = data[sequence_length:]
    # targets are the values right after each sequence
    data = data[:-sequence_length]
    # adjust data to match the size of targets
    dataset = tf.keras.preprocessing.timeseries_dataset_from_array(
        data=data,
        targets=targets,
        sequence_length=sequence_length,
        sequence_stride=1,
        shuffle=True,
        batch_size=batch_size
    )
    return dataset


csv_path = '../data/bitstampUSD_1-min_data_2012-01-01_to_2020-04-22.csv'
train_df, val_df, test_df = preprocess_data(csv_path)

sequence_length = 24  # Number of hours to look back for prediction
batch_size = 64

train_dataset = implement_dataset(train_df, sequence_length)
val_dataset = implement_dataset(val_df, sequence_length)
test_dataset = implement_dataset(test_df, sequence_length)
# You might need targets for testing as well

model = keras.Sequential([
    keras.layers.LSTM(64, input_shape=(sequence_length,
                                       train_df.shape[1]),
                      return_sequences=True),
    keras.layers.LSTM(32),
    keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse')

model.fit(train_dataset, validation_data=val_dataset, epochs=10)

mse = model.evaluate(test_dataset)
print(f"Test final value: {mse}")
