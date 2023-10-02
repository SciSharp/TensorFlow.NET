using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Tensorflow.Common.Types;
using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Keras.Engine;
using Tensorflow.Keras.Layers;
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
        public void TrainLSTMWithMnist()
        {
            var input = keras.Input((784));
            var x = keras.layers.Reshape((28, 28)).Apply(input);
            x = keras.layers.LSTM(50, return_sequences: true).Apply(x);
            x = keras.layers.LSTM(100).Apply(x);
            var output = keras.layers.Dense(10, activation: "softmax").Apply(x);

            var model = keras.Model(input, output);
            model.summary();
            model.compile(keras.optimizers.Adam(), keras.losses.CategoricalCrossentropy(), new string[] { "accuracy" });

            var data_loader = new MnistModelLoader();
            var dataset = data_loader.LoadAsync(new ModelLoadSetting
            {
                TrainDir = "mnist",
                OneHot = true,
                ValidationSize = 55000,
            }).Result;
            var sample_weight = np.ones(((int)dataset.Train.Data.shape[0]));
            model.fit(dataset.Train.Data, dataset.Train.Labels, batch_size: 16, epochs: 1, sample_weight:sample_weight);
        }

        [TestMethod]
        public void SimpleRNN()
        {
            var input = keras.Input((784));
            var x = keras.layers.Reshape((28, 28)).Apply(input);
            x = keras.layers.SimpleRNN(10).Apply(x);
            var output = keras.layers.Dense(10, activation: "softmax").Apply(x);

            var model = keras.Model(input, output);
            model.summary();
            model.compile(keras.optimizers.Adam(), keras.losses.CategoricalCrossentropy(), new string[] { "accuracy" });

            var data_loader = new MnistModelLoader();
            var dataset = data_loader.LoadAsync(new ModelLoadSetting
            {
                TrainDir = "mnist",
                OneHot = false,
                ValidationSize = 58000,
            }).Result;

            model.fit(dataset.Train.Data, dataset.Train.Labels, batch_size: 16, epochs: 2);
        }

        [TestMethod]
        public void RNNForSimpleRNNCell()
        {
            var inputs = tf.random.normal((32, 10, 8));
            var cell = tf.keras.layers.SimpleRNNCell(10, dropout: 0.5f, recurrent_dropout: 0.5f);
            var rnn = tf.keras.layers.RNN(cell: cell);
            var cgf = rnn.get_config();
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
        public void GRUCell()
        {
            var inputs = tf.random.normal((32, 10, 8));
            var rnn = tf.keras.layers.RNN(tf.keras.layers.GRUCell(4));
            var output = rnn.Apply(inputs);
            Assert.AreEqual((32, 4), output.shape);
            rnn = tf.keras.layers.RNN(tf.keras.layers.GRUCell(4, reset_after:false, use_bias:false));
            output = rnn.Apply(inputs);
            Assert.AreEqual((32, 4), output.shape);

        }

        [TestMethod]
        public void GRU()
        {
            var inputs = tf.ones((32, 10, 8));
            var gru = tf.keras.layers.GRU(4);
            var output = gru.Apply(inputs);
            Assert.AreEqual((32, 4), output.shape);
        }

        [TestMethod]
        public void Bidirectional()
        {
            var bi = tf.keras.layers.Bidirectional(keras.layers.LSTM(10, return_sequences:true));
            var inputs = tf.random.normal((32, 10, 8));
            var outputs = bi.Apply(inputs);
            Assert.AreEqual((32, 10, 20), outputs.shape);
        }
    }
}
