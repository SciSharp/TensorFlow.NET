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
            var model = tf.keras.models.Sequential();
            model.add(tf.keras.Input(shape: 16));
        }

        /// <summary>
        /// https://www.tensorflow.org/api_docs/python/tf/keras/layers/Embedding
        /// </summary>
        [TestMethod]
        public void Embedding()
        {
            var model = new Sequential();
            var layer = tf.keras.layers.Embedding(1000, 64, input_length: 10);
            model.add(layer);
            // the model will take as input an integer matrix of size (batch,
            // input_length).
            // the largest integer (i.e. word index) in the input should be no larger
            // than 999 (vocabulary size).
            // now model.output_shape == (None, 10, 64), where None is the batch
            // dimension.
            var input_array = np.random.randint(1000, size: (32, 10));
            // model.compile("rmsprop", "mse");
            // output_array = model.predict(input_array)
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
