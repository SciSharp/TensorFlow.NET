using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using Tensorflow;
using Tensorflow.Operations.Initializers;
using static Tensorflow.Binding;
using static Tensorflow.KerasApi;

namespace TensorFlowNET.Keras.UnitTest
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
            var model = keras.Sequential();
            model.add(keras.Input(shape: 16));
        }

        [TestMethod]
        public void Functional()
        {
            var layers = keras.layers;

            var inputs = keras.Input(shape: 784);
            Assert.AreEqual((-1, 784), inputs.TensorShape);

            var dense = layers.Dense(64, activation: keras.activations.Relu);
            var x = dense.Apply(inputs);

            x = layers.Dense(64, activation: keras.activations.Relu).Apply(x);
            var outputs = layers.Dense(10).Apply(x);

            var model = keras.Model(inputs, outputs, name: "mnist_model");
            model.summary();
        }

        /// <summary>
        /// Custom layer test, used in Dueling DQN
        /// </summary>
        [TestMethod, Ignore]
        public void TensorFlowOpLayer()
        {
            var layers = keras.layers;
            var inputs = layers.Input(shape: 24);
            var x = layers.Dense(128, activation: "relu").Apply(inputs);
            var value = layers.Dense(24).Apply(x);
            var adv = layers.Dense(1).Apply(x);

            var mean = adv - tf.reduce_mean(adv, axis: 1, keepdims: true);
            adv = layers.Subtract().Apply((adv, mean));
            var outputs = layers.Add().Apply((value, adv));
            var model = keras.Model(inputs, outputs);
            model.compile(optimizer: keras.optimizers.RMSprop(0.001f),
                          loss: keras.losses.MeanSquaredError(),
                          metrics: new[] { "acc" });
            model.summary();
            Assert.AreEqual(model.Layers.Count, 8);
            var result = model.predict(tf.constant(np.arange(24).astype(np.float32)[np.newaxis, Slice.All]));
            Assert.AreEqual(result.shape, new TensorShape(1, 24));
            model.fit(np.arange(24).astype(np.float32)[np.newaxis, Slice.All], np.arange(24).astype(np.float32)[np.newaxis, Slice.All], verbose: 0);
        }

        /// <summary>
        /// https://www.tensorflow.org/api_docs/python/tf/keras/layers/Embedding
        /// </summary>
        [TestMethod]
        public void Embedding_Simple()
        {
            var emb = keras.layers.Embedding(256, 12, input_length: 4);
            var input_array = np.arange(12).reshape(3, 4).astype(np.float32);
            var output = emb.Apply(input_array);
            Assert.AreEqual(new TensorShape(3, 4, 12), output.shape);
        }

        /// <summary>
        /// https://www.tensorflow.org/api_docs/python/tf/keras/layers/Embedding
        /// </summary>
        [TestMethod]
        [Ignore]
        public void Embedding()
        {
            var model = keras.Sequential();
            var layer = keras.layers.Embedding(7, 2, input_length: 4);
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
            // model.compile("rmsprop", "mse");
            var output_array = model.predict(input_array);
            Assert.AreEqual((32, 10, 64), output_array.shape);
        }

        /// <summary>
        /// https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense
        /// </summary>
        [TestMethod]
        public void Dense()
        {
            // Create a `Sequential` model and add a Dense layer as the first layer.
            var model = keras.Sequential();
            model.add(keras.Input(shape: 16));
            model.add(keras.layers.Dense(32, activation: keras.activations.Relu));
            // Now the model will take as input arrays of shape (None, 16)
            // and output arrays of shape (None, 32).
            // Note that after the first layer, you don't need to specify
            // the size of the input anymore:
            model.add(keras.layers.Dense(32));
            Assert.AreEqual((-1, 32), model.output_shape);
        }

        [TestMethod]
        [Ignore]
        public void SimpleRNN()
        {
            var inputs = np.random.rand(32, 10, 8).astype(np.float32);
            var simple_rnn = keras.layers.SimpleRNN(4);
            var output = simple_rnn.Apply(inputs);
            Assert.AreEqual((32, 4), output.shape);
        }

        [TestMethod]
        public void Resizing()
        {
            var inputs = tf.random.uniform((10, 32, 32, 3));
            var layer = keras.layers.preprocessing.Resizing(16, 16);
            var output = layer.Apply(inputs);
            Assert.AreEqual((10, 16, 16, 3), output.shape);
        }
    }
}
