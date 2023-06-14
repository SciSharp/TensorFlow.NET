using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Tensorflow.Common.Types;
using Tensorflow.Keras.Engine;
using Tensorflow.Keras.Layers.Rnn;
using Tensorflow.Keras.Saving;
using Tensorflow.NumPy;
using Tensorflow.Train;
using static Tensorflow.Binding;
using static Tensorflow.KerasApi;

namespace Tensorflow.Keras.UnitTest.Layers
{
    [TestClass]
    public class Rnn
    {
        [TestMethod]
        public void SimpleRNNCell()
        {
            var cell = tf.keras.layers.SimpleRNNCell(64, dropout: 0.5f, recurrent_dropout: 0.5f);
            var h0 = new Tensors { tf.zeros(new Shape(4, 64)) };
            var x = tf.random.normal((4, 100));
            var (y, h1) = cell.Apply(inputs: x, states: h0);
            var h2 = h1;
            Assert.AreEqual((4, 64), y.shape);
            Assert.AreEqual((4, 64), h2[0].shape);
        }

        [TestMethod]
        public void StackedRNNCell()
        {
            var inputs = tf.ones((32, 10));
            var states = new Tensors { tf.zeros((32, 4)), tf.zeros((32, 5)) };
            var cells = new IRnnCell[] { tf.keras.layers.SimpleRNNCell(4), tf.keras.layers.SimpleRNNCell(5) };
            var stackedRNNCell = tf.keras.layers.StackedRNNCells(cells);
            var (output, state) = stackedRNNCell.Apply(inputs, states);
            Console.WriteLine(output);
            Console.WriteLine(state.shape);
            Assert.AreEqual((32, 5), output.shape);
            Assert.AreEqual((32, 4), state[0].shape);
        }

        [TestMethod]
        public void LSTMCell()
        {
            var inputs = tf.ones((2, 100));
            var states = new Tensors { tf.zeros((2, 4)), tf.zeros((2, 4)) };
            var rnn = tf.keras.layers.LSTMCell(4);
            var (output, new_states) = rnn.Apply(inputs, states);
            Assert.AreEqual((2, 4), output.shape);
            Assert.AreEqual((2, 4), new_states[0].shape);
        }

        [TestMethod]
        public void SimpleRNN()
        {
            //var inputs = np.arange(6 * 10 * 8).reshape((6, 10, 8)).astype(np.float32);
            ///*var simple_rnn = keras.layers.SimpleRNN(4);
            //var output = simple_rnn.Apply(inputs);
            //Assert.AreEqual((32, 4), output.shape);*/

            //var simple_rnn = tf.keras.layers.SimpleRNN(4, return_sequences: true, return_state: true);
            //var (whole_sequence_output, final_state) = simple_rnn.Apply(inputs);
            //Assert.AreEqual((6, 10, 4), whole_sequence_output.shape);
            //Assert.AreEqual((6, 4), final_state.shape);

            var inputs = keras.Input(shape: (10, 8));
            var x = keras.layers.SimpleRNN(4).Apply(inputs);
            var output = keras.layers.Dense(10).Apply(x);
            var model = keras.Model(inputs, output);
            model.summary();
        }
        [TestMethod]
        public void RNNForSimpleRNNCell()
        {
            var inputs = tf.random.normal((32, 10, 8));
            var cell = tf.keras.layers.SimpleRNNCell(10, dropout: 0.5f, recurrent_dropout: 0.5f);
            var rnn = tf.keras.layers.RNN(cell: cell);
            var output = rnn.Apply(inputs);
            Assert.AreEqual((32, 10), output.shape);

        }
        [TestMethod]
        public void RNNForStackedRNNCell()
        {
            var inputs = tf.random.normal((32, 10, 8));
            var cells = new IRnnCell[] { tf.keras.layers.SimpleRNNCell(4), tf.keras.layers.SimpleRNNCell(5) };
            var stackedRNNCell = tf.keras.layers.StackedRNNCells(cells);
            var rnn = tf.keras.layers.RNN(cell: stackedRNNCell);
            var output = rnn.Apply(inputs);
            Assert.AreEqual((32, 5), output.shape);
        }

        [TestMethod]
        public void RNNForLSTMCell()
        {
            var inputs = tf.ones((5, 10, 8));
            var rnn = tf.keras.layers.RNN(tf.keras.layers.LSTMCell(4));
            var output = rnn.Apply(inputs);
            Console.WriteLine($"output: {output}");
            Assert.AreEqual((5, 4), output.shape);
        }

        [TestMethod]
        public void MyTest()
        {
            var a = tf.zeros((2, 3));
            var b = tf.ones_like(a);
            var c = tf.ones((3,4));

            var d = new Tensors { a, b, c };
            var (A, BC) = d;
            Console.WriteLine($"A:{A}");
            Console.WriteLine($"BC:{BC}");
        }

    }
}
