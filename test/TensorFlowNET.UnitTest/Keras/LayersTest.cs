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
    [TestClass, Ignore]
    public class LayersTest : GraphModeTestBase
    {
        [TestMethod]
        public void Sequential()
        {
            var model = tf.keras.models.Sequential();
            model.add(tf.keras.Input(shape: 16));
        }

        [TestMethod]
        public void Embedding()
        {
            var model = new Sequential();
            model.add(new Embedding(1000, 64, input_length: 10));
            // the model will take as input an integer matrix of size (batch,
            // input_length).
            // the largest integer (i.e. word index) in the input should be no larger
            // than 999 (vocabulary size).
            // now model.output_shape == (None, 10, 64), where None is the batch
            // dimension.
            var input_array = np.random.randint(1000, size: (32, 10));
            model.compile("rmsprop", "mse");
        }

        [TestMethod]
        public void Dense()
        {
            var model = tf.keras.Sequential();
            var dense_layer = tf.keras.layers.Dense(5, input_shape: 3);
            model.add(dense_layer);
        }
    }
}
