import os
import numpy as np
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
print(tf.__version__)
# https://playground.tensorflow.org/
# tf.compat.v1.enable_eager_execution()
# tf.debugging.set_log_device_placement(True);
tf.config.run_functions_eagerly(True)

x = np.array([[ 0, 0 ], [ 0, 1 ], [ 1, 0 ], [ 1, 1 ]])
y = np.array([[ 0 ], [ 1 ], [ 1 ], [ 0 ] ])

model = tf.keras.Sequential()
model.add(tf.keras.Input(2))
model.add(tf.keras.layers.Dense(32, "relu"))
model.add(tf.keras.layers.Dense(1, "sigmoid"))
model.compile(optimizer = tf.keras.optimizers.Adam(),
    loss = tf.keras.losses.MeanSquaredError(),
    metrics = ["accuracy"])
model.fit(x, y, 1, 100)
result = model.evaluate(x, y)
print(model.predict(x, 4))