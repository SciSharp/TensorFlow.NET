using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Keras;
using Tensorflow.NumPy;
using static Tensorflow.Binding;
using static Tensorflow.KerasApi;

namespace Tensorflow
{
    public class SimpleRnnTest
    {
        public void Run()
        {
            tf.keras = new KerasInterface();
            var inputs = np.random.random((32, 10, 8)).astype(np.float32);
            var simple_rnn = tf.keras.layers.SimpleRNN(4);
            var output = simple_rnn.Apply(inputs);  // The output has shape `[32, 4]`.
            if (output.shape == (32, 4))
            {

            }
            /*simple_rnn = tf.keras.layers.SimpleRNN(
                4, return_sequences = True, return_state = True)

            # whole_sequence_output has shape `[32, 10, 4]`.
            # final_state has shape `[32, 4]`.
            whole_sequence_output, final_state = simple_rnn(inputs)*/
        }
    }
}
