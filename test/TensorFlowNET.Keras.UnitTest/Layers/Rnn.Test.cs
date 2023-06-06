using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Tensorflow.NumPy;
using static Tensorflow.Binding;

namespace Tensorflow.Keras.UnitTest.Layers
{
    [TestClass]
    public class Rnn
    {
        [TestMethod]
        public void SimpleRNN()
        {
            var inputs = np.arange(6 * 10 * 8).reshape((6, 10, 8)).astype(np.float32);
            /*var simple_rnn = keras.layers.SimpleRNN(4);
            var output = simple_rnn.Apply(inputs);
            Assert.AreEqual((32, 4), output.shape);*/
            var simple_rnn = tf.keras.layers.SimpleRNN(4, return_sequences: true, return_state: true);
            var (whole_sequence_output, final_state) = simple_rnn.Apply(inputs);
            Console.WriteLine(whole_sequence_output);
            Console.WriteLine(final_state);
        }
    }
}
