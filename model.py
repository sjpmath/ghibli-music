import tensorflow as tf
import numpy as np
from dataset import seq_length

# for step and duration: encourage output positive values
def mse_with_positive_pressure(y_true: tf.Tensor, y_pred: tf.Tensor):
  mse = (y_true-y_pred)**2
  positive_pressure = 100*tf.maximum(-y_pred, 0.0)
  return tf.reduce_mean(mse+positive_pressure)
  # loss is higher the more negative y_pred is

input_shape = (seq_length, 3)
learning_rate = 0.005
inputs = tf.keras.Input(input_shape)
x = tf.keras.layers.LSTM(128)(inputs)

outputs = {
    'pitch': tf.keras.layers.Dense(128, name='pitch')(x),
    'step': tf.keras.layers.Dense(1, name='step')(x),
    'duration': tf.keras.layers.Dense(1, name='duration')(x)
}

model = tf.keras.Model(inputs, outputs)

loss = {
    'pitch': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    'step': mse_with_positive_pressure,
    'duration': mse_with_positive_pressure,
}

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)