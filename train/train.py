#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
import struct
import chess
import functools
from enum import Enum
from enum import IntFlag
from datetime import datetime

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from halfkp import get_halfkp_indeces
import lichess_pgn_data_generator

FEATURE_TRANSFORMER_HALF_DIMENSIONS = 256
DENSE_LAYERS_WIDTH = 32

def build_model_inputs():
  # return keras.Input(shape=(41024,), sparse=True), keras.Input(shape=(41024,), sparse=True)
  return keras.Input(shape=(41024,)), keras.Input(shape=(41024,))

def build_feature_transformer(inputs1, inputs2):
  ft_dense_layer = layers.Dense(FEATURE_TRANSFORMER_HALF_DIMENSIONS, name='feature_transformer')
  clipped_relu = layers.ReLU(max_value=127)
  transformed1 = clipped_relu(ft_dense_layer(inputs1))
  transformed2 = clipped_relu(ft_dense_layer(inputs2))
  return tf.keras.layers.Concatenate()([transformed1, transformed2])

def build_hidden_layers(inputs):
  hidden_layer_1 = layers.Dense(DENSE_LAYERS_WIDTH, name='hidden_layer_1')
  hidden_layer_2 = layers.Dense(DENSE_LAYERS_WIDTH, name='hidden_layer_2')
  activation_1 = layers.ReLU(max_value=127.0)
  activation_2 = layers.ReLU(max_value=127.0)
  return activation_2(hidden_layer_2(activation_1(hidden_layer_1(inputs))))

def build_output_layer(inputs):
  output_layer = layers.Dense(1, name='output_layer')
  return output_layer(inputs)

def build_model():
  inputs1, inputs2 = build_model_inputs()
  outputs = build_output_layer(build_hidden_layers(build_feature_transformer(inputs1, inputs2)))
  return keras.Model(inputs=[inputs1, inputs2], outputs=outputs)

dataset = tf.data.Dataset.from_generator(
  lichess_pgn_data_generator.gen, args=['../train/lichess_db_standard_rated_2017-02.pgn'],
  output_types=((tf.float32, tf.float32), tf.float32)
)

model = build_model()

logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(
  log_dir=logdir,
  histogram_freq=200
)

opt = keras.optimizers.Adam(learning_rate=0.1)
model.compile(
  optimizer=opt,
  loss='mse',
  metrics=[
    tf.keras.metrics.MeanSquaredError(),
    tf.keras.metrics.MeanAbsoluteError()
  ]
)

keras.utils.plot_model(model, show_shapes=True, show_layer_names=True, to_file='model.png')

model.fit(
  dataset.batch(32),
  callbacks=[tensorboard_callback]
)
