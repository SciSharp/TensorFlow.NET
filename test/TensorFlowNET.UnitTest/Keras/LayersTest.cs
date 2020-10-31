using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Keras.Engine;
using Tensorflow.Keras.Layers;
using NumSharp;
using Tensorflow.UnitTest;
using static Tensorflow.Binding;

namespace TensorFlowNET.UnitTest.Keras
{
    /// <summary>
    /// https://www.tensorflow.org/versions/r2.3/api_docs/python/tf/keras/layers
    /// </summary>
    [TestClass]
    public class LayersTest : EagerModeTestBase
    {
        
        [TestMethod]
        public void Sequential()
        {
            var model = tf.keras.Sequential();
            model.add(tf.keras.Input(shape: 16));
        }

        [TestMethod]
        public void Functional()
        {
            var inputs = keras.Input(shape: 784);
            Assert.AreEqual((None, 784), inputs.TensorShape);

            var dense = layers.Dense(64, activation: tf.keras.activations.Relu);
            var x = dense.Apply(inputs);

            x = layers.Dense(64, activation: tf.keras.activations.Relu).Apply(x);
            var outputs = layers.Dense(10).Apply(x);

            var model = keras.Model(inputs, outputs, name: "mnist_model");
            model.summary();
        }

        /// <summary>
        /// https://www.tensorflow.org/api_docs/python/tf/keras/layers/Embedding
        /// </summary>
        [TestMethod, Ignore]
        public void Embedding()
        {
            var model = tf.keras.Sequential();
            var layer = tf.keras.layers.Embedding(7, 2, input_length: 4);
            model.add(layer);
            // the model will take as input an integer matrix of size (batch,
            // input_length).
            // the largest integer (i.e. word index) in the input should be no larger
            // than 999 (vocabulary size).
            // now model.output_shape == (None, 10, 64), where None is the batch
            // dimension.
            var input_array = np.array(new int[,]
            {
                { 1, 2, 3, 4 },
                { 2, 3, 4, 5 },
                { 3, 4, 5, 6 }
            });
            model.compile("rmsprop", "mse");
            var output_array = model.predict(input_array);
            Assert.AreEqual((32, 10, 64), output_array.TensorShape);
        }

        /// <summary>
        /// https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense
        /// </summary>
        [TestMethod]
        public void Dense()
        {
            // Create a `Sequential` model and add a Dense layer as the first layer.
            var model = tf.keras.Sequential();
            model.add(tf.keras.Input(shape: 16));
            model.add(tf.keras.layers.Dense(32, activation: tf.keras.activations.Relu));
            // Now the model will take as input arrays of shape (None, 16)
            // and output arrays of shape (None, 32).
            // Note that after the first layer, you don't need to specify
            // the size of the input anymore:
            model.add(tf.keras.layers.Dense(32));
            Assert.AreEqual((-1, 32), model.output_shape);
        }

        [TestMethod]
        public void SimpleRNN()
        {

        }
    }
}
