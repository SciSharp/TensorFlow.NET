import numpy as np
import tensorflow as tf
import tensorflow.experimental.numpy as tnp

# tf.experimental.numpy
inputs = np.arange(6 * 10 * 8).reshape([6, 10, 8]).astype(np.float32)
# simple_rnn = tf.keras.layers.SimpleRNN(4)

# output = simple_rnn(inputs)  # The output has shape `[6, 4]`.

simple_rnn = tf.keras.layers.SimpleRNN(4, return_sequences=True, return_state=True)

# whole_sequence_output has shape `[6, 10, 4]`.
# final_state has shape `[6, 4]`.
whole_sequence_output, final_state = simple_rnn(inputs)
print(whole_sequence_output)
print(final_state)