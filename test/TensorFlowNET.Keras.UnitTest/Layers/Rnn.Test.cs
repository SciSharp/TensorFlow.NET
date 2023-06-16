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
            //var cell = tf.keras.layers.SimpleRNNCell(64, dropout: 0.5f, recurrent_dropout: 0.5f);
            //var h0 = new Tensors { tf.zeros(new Shape(4, 64)) };
            //var x = tf.random.normal((4, 100));
            //var (y, h1) = cell.Apply(inputs: x, states: h0);
            //var h2 = h1;
            //Assert.AreEqual((4, 64), y.shape);
            //Assert.AreEqual((4, 64), h2[0].shape);

            //var model = keras.Sequential(new List<ILayer>
            //{
            //  keras.layers.InputLayer(input_shape: (4,100)),
            //  keras.layers.SimpleRNNCell(64)
            //});
            //model.summary();

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

            model.compile(keras.optimizers.Adam(), keras.losses.SparseCategoricalCrossentropy());
            var datax = np.ones((16, 10, 8), dtype: dtypes.float32);
            var datay = np.ones((16));
            model.fit(datax, datay, epochs: 20);
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
        public void WlzTest()
        {
            long[] b = { 1, 2, 3 };
            
            Shape a = new Shape(Unknown).concatenate(b);
            Console.WriteLine(a);

        }


    }
}
