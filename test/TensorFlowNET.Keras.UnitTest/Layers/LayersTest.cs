using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Linq;
using Tensorflow.NumPy;
using static Tensorflow.Binding;
using static Tensorflow.KerasApi;

namespace Tensorflow.Keras.UnitTest.Layers
{
    /// <summary>
    /// https://www.tensorflow.org/versions/r2.3/api_docs/python/tf/keras/layers
    /// </summary>
    [TestClass]
    public class LayersTest : EagerModeTestBase
    {
        [TestMethod]
        public void AveragePooling2D()
        {
            var x = tf.constant(new float[,]
            {
                { 1, 2, 3 },
                { 4, 5, 6 },
                { 7, 8, 9 }
            });
            x = tf.reshape(x, (1, 3, 3, 1));
            var avg_pool_2d = keras.layers.AveragePooling2D(pool_size: (2, 2),
                strides: (1, 1), padding: "valid");
            Tensor avg = avg_pool_2d.Apply(x);
            Assert.AreEqual((1, 2, 2, 1), avg.shape);
            Equal(new float[] { 3, 4, 6, 7 }, avg.ToArray<float>());
        }

        [TestMethod]
        public void InputLayer()
        {
            var model = keras.Sequential(new List<ILayer>
            {
              keras.layers.InputLayer(input_shape: 4),
              keras.layers.Dense(8)
            });
            model.compile(optimizer: keras.optimizers.RMSprop(0.001f),
                loss: keras.losses.MeanSquaredError(),
                metrics: new[] { "accuracy" });
            model.fit(np.zeros((10, 4), dtype: tf.float32), np.ones((10, 8), dtype: tf.float32));
        }

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
            Assert.AreEqual((-1, 784), inputs.shape);

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
            Assert.AreEqual(result.shape, new Shape(1, 24));
            model.fit(np.arange(24).astype(np.float32)[np.newaxis, Slice.All], np.arange(24).astype(np.float32)[np.newaxis, Slice.All], verbose: 0);
        }

        /// <summary>
        /// https://www.tensorflow.org/api_docs/python/tf/keras/layers/Embedding
        /// </summary>
        [TestMethod]
        public void Embedding()
        {
            var model = keras.Sequential();
            var layer = keras.layers.Embedding(1000, 64, input_length: 10);
            model.add(layer);
            var input_array = np.random.randint(1000, size: (32, 10));
            model.compile("rmsprop", "mse", new[] { "accuracy" });
            var output_array = model.predict(input_array);
            Assert.AreEqual((32, 10, 64), output_array.shape);
        }
        [TestMethod]
        public void EmbeddingGrad()
        {
            var inputs = keras.layers.Input(shape: new[] { 32, 10 });
            var outputs = keras.layers.Embedding(1000, 64, input_length: 10).Apply(inputs);
            var model = keras.Model(inputs: inputs, outputs: outputs);
            var input_array = np.random.randint(1000, size: (1, 32, 10));
            var output_array = np.random.random(size: (1, 32, 10, 64));
            model.compile("rmsprop", "mse", new[] { "accuracy" });
            model.fit(input_array, output_array);
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
        public void EinsumDense()
        {
            var ed = keras.layers.EinsumDense(
                equation: "...b,bc->...c",
                output_shape: 4,
                bias_axes: "c",
                bias_initializer: tf.constant_initializer(0.03),
                kernel_initializer: tf.constant_initializer(0.5)
            );
            var inp = np.array(new[,] { { 1f, 2f }, { 3f, 4f } });
            var expected_output = np.array(new[,] {{1.53f, 1.53f, 1.53f, 1.53f },
                                { 3.53f, 3.53f, 3.53f, 3.53f }});
            var actual_output = ed.Apply(inp)[0].numpy();
            Assert.AreEqual(expected_output, actual_output);
        }

        [TestMethod]
        public void Resizing()
        {
            var inputs = tf.random.uniform((10, 32, 32, 3));
            var layer = keras.layers.preprocessing.Resizing(16, 16);
            var output = layer.Apply(inputs);
            Assert.AreEqual((10, 16, 16, 3), output.shape);
        }

        [TestMethod]
        public void LayerNormalization()
        {
            var inputs = tf.constant(np.arange(10).reshape((5, 2)) * 10, dtype: tf.float32);
            var layer = keras.layers.LayerNormalization(axis: 1);
            Tensor output = layer.Apply(inputs);
            Assert.AreEqual((5, 2), output.shape);
            Assert.IsTrue(output[0].numpy().Equals(new[] { -0.99998f, 0.99998f }));

            // test_layernorm_weights
            Assert.AreEqual(len(layer.TrainableWeights), 2);
            Assert.AreEqual(len(layer.Weights), 2);

            var beta = layer.Weights.Where(x => x.Name.StartsWith("beta")).Single();
            var gamma = layer.Weights.Where(x => x.Name.StartsWith("gamma")).Single();

            // correctness_test
            layer = keras.layers.LayerNormalization(axis: -1, epsilon: (float) 1e-12);
            var x = np.random.normal(loc: 5.0f, scale: 10.0f, size: (1000, 2, 2, 2)).astype(tf.float32);

            output = layer.Apply(x);

            var y = (output - beta.numpy()) / gamma.numpy();

            var y_mean = np.mean(y.numpy());
            var y_std = np.sqrt(np.sum(np.power(y.numpy() - np.mean(y.numpy()), 2)) / 8000);
            Assert.IsTrue(tf.greater(np.array(0.1f), tf.abs(y_std - 1.0)).ToArray<bool>()[0]);
            Assert.IsTrue(tf.greater(np.array(0.1f), tf.abs(y_mean)).ToArray<bool>()[0]);
        }

        /// <summary>
        /// https://www.tensorflow.org/api_docs/python/tf/keras/layers/Normalization
        /// </summary>
        [TestMethod]
        public void Normalization()
        {
            // Calculate a global mean and variance by analyzing the dataset in adapt().
            var adapt_data = np.array(new[] { 1f, 2f, 3f, 4f, 5f });
            var input_data = np.array(new[] { 1f, 2f, 3f });
            var layer = tf.keras.layers.Normalization(axis: null);
            layer.adapt(adapt_data);
            var x = layer.Apply(input_data);
            Assert.AreEqual(x.numpy(), new[] { -1.4142135f, -0.70710677f, 0f });

            // Calculate a mean and variance for each index on the last axis.
            adapt_data = np.array(new[,]
            {
                { 0, 7, 4 },
                { 2, 9, 6 },
                { 0, 7, 4 },
                { 2, 9, 6 }
            }, dtype: tf.float32);
            input_data = np.array(new[,] { { 0, 7, 4 } }, dtype: tf.float32);
            layer = tf.keras.layers.Normalization(axis: -1);
            layer.adapt(adapt_data);
            x = layer.Apply(input_data);
            Equal(x.numpy().ToArray<float>(), new[] { -1f, -1f, -1f });

            // Pass the mean and variance directly.
            input_data = np.array(new[,] { { 1f }, { 2f }, { 3f } }, dtype: tf.float32);
            layer = tf.keras.layers.Normalization(mean: 3f, variance: 2f);
            x = layer.Apply(input_data);
            Equal(x.numpy().ToArray<float>(), new[] { -1.4142135f, -0.70710677f, 0f });

            // Use the layer to de-normalize inputs (after adapting the layer).
            adapt_data = np.array(new[,]
            {
                { 0, 7, 4 },
                { 2, 9, 6 },
                { 0, 7, 4 },
                { 2, 9, 6 }
            }, dtype: tf.float32);
            input_data = np.array(new[,] { { 1, 2, 3 } }, dtype: tf.float32);
            layer = tf.keras.layers.Normalization(axis: -1, invert: true);
            layer.adapt(adapt_data);
            x = layer.Apply(input_data);
            Equal(x.numpy().ToArray<float>(), new[] { -2f, -10f, -8f });
        }

        /// <summary>
        /// https://www.tensorflow.org/api_docs/python/tf/keras/layers/CategoryEncoding
        /// </summary>
        [TestMethod]
        public void CategoryEncoding()
        {
            // one-hot
            var inputs = np.array(new[] { 3, 2, 0, 1 });
            var layer = tf.keras.layers.CategoryEncoding(4);

            Tensor output = layer.Apply(inputs);
            Assert.AreEqual((4, 4), output.shape);
            Assert.IsTrue(output[0].numpy().Equals(new[] { 0, 0, 0, 1f }));
            Assert.IsTrue(output[1].numpy().Equals(new[] { 0, 0, 1, 0f }));
            Assert.IsTrue(output[2].numpy().Equals(new[] { 1, 0, 0, 0f }));
            Assert.IsTrue(output[3].numpy().Equals(new[] { 0, 1, 0, 0f }));

            // multi-hot
            inputs = np.array(new[,]
            {
                { 0, 1 },
                { 0, 0 },
                { 1, 2 },
                { 3, 1 }
            });
            layer = tf.keras.layers.CategoryEncoding(4, output_mode: "multi_hot");
            output = layer.Apply(inputs);
            Assert.IsTrue(output[0].numpy().Equals(new[] { 1, 1, 0, 0f }));
            Assert.IsTrue(output[1].numpy().Equals(new[] { 1, 0, 0, 0f }));
            Assert.IsTrue(output[2].numpy().Equals(new[] { 0, 1, 1, 0f }));
            Assert.IsTrue(output[3].numpy().Equals(new[] { 0, 1, 0, 1f }));

            // using weighted inputs in "count" mode
            inputs = np.array(new[,]
            {
                { 0, 1 },
                { 0, 0 },
                { 1, 2 },
                { 3, 1 }
            });
            var weights = np.array(new[,]
            {
                { 0.1f, 0.2f },
                { 0.1f, 0.1f },
                { 0.2f, 0.3f },
                { 0.4f, 0.2f }
            });
            layer = tf.keras.layers.CategoryEncoding(4, output_mode: "count", count_weights: weights);
            output = layer.Apply(inputs);
            Assert.IsTrue(output[0].numpy().Equals(new[] { 0.1f, 0.2f, 0f, 0f }));
            Assert.IsTrue(output[1].numpy().Equals(new[] { 0.2f, 0f, 0f, 0f }));
            Assert.IsTrue(output[2].numpy().Equals(new[] { 0f, 0.2f, 0.3f, 0f }));
            Assert.IsTrue(output[3].numpy().Equals(new[] { 0f, 0.2f, 0f, 0.4f }));
        }
    }
}
