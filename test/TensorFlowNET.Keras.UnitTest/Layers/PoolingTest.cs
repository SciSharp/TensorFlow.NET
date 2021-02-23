using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using System.Linq;
using Tensorflow;
using static Tensorflow.Binding;
using static Tensorflow.KerasApi;

namespace TensorFlowNET.Keras.UnitTest
{
    /// <summary>
    /// https://www.tensorflow.org/versions/r2.3/api_docs/python/tf/keras/layers
    /// </summary>
    [TestClass]
    public class PoolingTest : EagerModeTestBase
    {
        private NDArray input_array_1D = np.array(new float[,,]
            {
                {{1,2,3,3,3},{1,2,3,3,3},{1,2,3,3,3}},
                {{4,5,6,3,3},{4,5,6,3,3},{4,5,6,3,3}},
                {{7,8,9,3,3},{7,8,9,3,3},{7,8,9,3,3}},
                {{7,8,9,3,3},{7,8,9,3,3},{7,8,9,3,3}}
            });

        private NDArray input_array_2D = np.array(new float[,,,]
            {{
                {{1,2,3,3,3},{1,2,3,3,3},{1,2,3,3,3}},
                {{4,5,6,3,3},{4,5,6,3,3},{4,5,6,3,3}},
                },{
                {{7,8,9,3,3},{7,8,9,3,3},{7,8,9,3,3}},
                {{7,8,9,3,3},{7,8,9,3,3},{7,8,9,3,3}}
                },{
                {{1,2,3,3,3},{1,2,3,3,3},{1,2,3,3,3}},
                {{4,5,6,3,3},{4,5,6,3,3},{4,5,6,3,3}},
                },{
                {{7,8,9,3,3},{7,8,9,3,3},{7,8,9,3,3}},
                {{7,8,9,3,3},{7,8,9,3,3},{7,8,9,3,3}}
            }});

        [TestMethod]
        public void GlobalAverage1DPoolingChannelsLast()
        {
            var pool = keras.layers.GlobalAveragePooling1D();
            var y = pool.Apply(input_array_1D);

            Assert.AreEqual(4, y.shape[0]);
            Assert.AreEqual(5, y.shape[1]);

            var expected = np.array(new float[,]
            {
                {1,2,3,3,3},
                {4,5,6,3,3},
                {7,8,9,3,3},
                {7,8,9,3,3}
            });

            Assert.AreEqual(expected, y[0].numpy());
        }

        [TestMethod]
        public void GlobalAverage1DPoolingChannelsFirst()
        {
            var pool = keras.layers.GlobalAveragePooling1D(data_format: "channels_first");
            var y = pool.Apply(input_array_1D);

            Assert.AreEqual(4, y.shape[0]);
            Assert.AreEqual(3, y.shape[1]);

            var expected = np.array(new float[,]
            {
                {2.4f, 2.4f, 2.4f},
                {4.2f, 4.2f, 4.2f},
                {6.0f, 6.0f, 6.0f},
                {6.0f, 6.0f, 6.0f}
            });

            Assert.AreEqual(expected, y[0].numpy());
        }

        [TestMethod]
        public void GlobalAverage2DPoolingChannelsLast()
        {
            var pool = keras.layers.GlobalAveragePooling2D();
            var y = pool.Apply(input_array_2D);

            Assert.AreEqual(4, y.shape[0]);
            Assert.AreEqual(5, y.shape[1]);

            var expected = np.array(new float[,]
            {
                {2.5f, 3.5f, 4.5f, 3.0f, 3.0f},
                {7.0f, 8.0f, 9.0f, 3.0f, 3.0f},
                {2.5f, 3.5f, 4.5f, 3.0f, 3.0f},
                {7.0f, 8.0f, 9.0f, 3.0f, 3.0f}
            });

            Assert.AreEqual(expected, y[0].numpy());
        }

        [TestMethod]
        public void GlobalAverage2DPoolingChannelsFirst()
        {
            var pool = keras.layers.GlobalAveragePooling2D(data_format: "channels_first");
            var y = pool.Apply(input_array_2D);

            Assert.AreEqual(4, y.shape[0]);
            Assert.AreEqual(2, y.shape[1]);

            var expected = np.array(new float[,]
            {
                {2.4f, 4.2f},
                {6.0f, 6.0f},
                {2.4f, 4.2f},
                {6.0f, 6.0f}
            });

            Assert.AreEqual(expected, y[0].numpy());
        }

        [TestMethod]
        public void GlobalMax1DPoolingChannelsLast()
        {
            var pool = keras.layers.GlobalMaxPooling1D();
            var y = pool.Apply(input_array_1D);

            Assert.AreEqual(4, y.shape[0]);
            Assert.AreEqual(5, y.shape[1]);

            var expected = np.array(new float[,]
            {
                {1,2,3,3,3},
                {4,5,6,3,3},
                {7,8,9,3,3},
                {7,8,9,3,3}
            });

            Assert.AreEqual(expected, y[0].numpy());
        }

        [TestMethod]
        public void GlobalMax1DPoolingChannelsFirst()
        {
            var pool = keras.layers.GlobalMaxPooling1D(data_format: "channels_first");
            var y = pool.Apply(input_array_1D);

            Assert.AreEqual(4, y.shape[0]);
            Assert.AreEqual(3, y.shape[1]);

            var expected = np.array(new float[,]
            {
                {3.0f, 3.0f, 3.0f},
                {6.0f, 6.0f, 6.0f},
                {9.0f, 9.0f, 9.0f},
                {9.0f, 9.0f, 9.0f}
            });

            Assert.AreEqual(expected, y[0].numpy());
        }

        [TestMethod]
        public void GlobalMax2DPoolingChannelsLast()
        {
            var input_array_2D = np.array(new float[,,,]
            {{
                {{1,2,3,3,3},{1,2,3,3,3},{1,2,3,9,3}},
                {{4,5,6,3,3},{4,5,6,3,3},{4,5,6,3,3}},
                },{
                {{7,8,9,3,3},{7,8,9,3,3},{7,8,9,3,3}},
                {{7,8,9,3,3},{7,8,9,3,3},{7,8,9,3,3}}
                },{
                {{1,2,3,3,3},{1,2,3,3,3},{1,2,3,3,9}},
                {{4,5,6,3,3},{4,5,6,3,3},{4,5,6,3,3}},
                },{
                {{7,8,9,3,3},{7,8,9,3,3},{7,8,9,3,3}},
                {{7,8,9,3,3},{7,8,9,3,3},{7,8,9,3,3}}
            }});

            var pool = keras.layers.GlobalMaxPooling2D();
            var y = pool.Apply(input_array_2D);

            Assert.AreEqual(4, y.shape[0]);
            Assert.AreEqual(5, y.shape[1]);

            var expected = np.array(new float[,]
            {
                {4.0f, 5.0f, 6.0f, 9.0f, 3.0f},
                {7.0f, 8.0f, 9.0f, 3.0f, 3.0f},
                {4.0f, 5.0f, 6.0f, 3.0f, 9.0f},
                {7.0f, 8.0f, 9.0f, 3.0f, 3.0f}
            });

            Assert.AreEqual(expected, y[0].numpy());
        }

        [TestMethod]
        public void GlobalMax2DPoolingChannelsFirst()
        {
            var input_array_2D = np.array(new float[,,,]
            {{
                {{1,2,3,3,3},{1,2,3,3,3},{1,2,3,9,3}},
                {{4,5,6,3,3},{4,5,6,3,3},{4,5,6,3,3}},
                },{
                {{7,8,9,3,3},{7,8,9,3,3},{7,8,9,3,3}},
                {{7,8,9,3,3},{7,8,9,3,3},{7,8,9,3,3}}
                },{
                {{1,2,3,3,3},{1,2,3,3,3},{1,2,3,3,9}},
                {{4,5,6,3,3},{4,5,6,3,3},{4,5,6,3,3}},
                },{
                {{7,8,9,3,3},{7,8,9,3,3},{7,8,9,3,3}},
                {{7,8,9,3,3},{7,8,9,3,3},{7,8,9,3,3}}
            }});

            var pool = keras.layers.GlobalMaxPooling2D(data_format: "channels_first");
            var y = pool.Apply(input_array_2D);

            Assert.AreEqual(4, y.shape[0]);
            Assert.AreEqual(2, y.shape[1]);

            var expected = np.array(new float[,]
            {
                {9.0f, 6.0f},
                {9.0f, 9.0f},
                {9.0f, 6.0f},
                {9.0f, 9.0f}
            });

            Assert.AreEqual(expected, y[0].numpy());
        }

        [TestMethod, Ignore("There's an error generated from TF complaining about the shape of the pool. Needs further investigation.")]
        public void Max1DPoolingChannelsLast()
        {
            var x = input_array_1D;
            var pool = keras.layers.MaxPooling1D(pool_size:2, strides:1);
            var y = pool.Apply(x);

            Assert.AreEqual(4, y.shape[0]);
            Assert.AreEqual(2, y.shape[1]);
            Assert.AreEqual(5, y.shape[2]);

            var expected = np.array(new float[,,]
            {
                {{2.0f, 2.0f, 3.0f, 3.0f, 3.0f},
                 { 1.0f, 2.0f, 3.0f, 3.0f, 3.0f}},

                {{4.0f, 5.0f, 6.0f, 3.0f, 3.0f},
                 {4.0f, 5.0f, 6.0f, 3.0f, 3.0f}},

                {{7.0f, 8.0f, 9.0f, 3.0f, 3.0f},
                 {7.0f, 8.0f, 9.0f, 3.0f, 3.0f}},

                {{7.0f, 8.0f, 9.0f, 3.0f, 3.0f},
                 {7.0f, 8.0f, 9.0f, 3.0f, 3.0f}}
            });

            Assert.AreEqual(expected, y[0].numpy());
        }

        [TestMethod]
        public void Max2DPoolingChannelsLast()
        {
            var x = np.array(new float[,,,]
            {{
                {{1,2,3,3,3},{1,2,3,3,3},{1,2,3,9,3}},
                {{4,5,6,3,3},{4,5,6,3,3},{4,5,6,3,3}},
                },{
                {{7,8,9,3,3},{7,8,9,3,3},{7,8,9,3,3}},
                {{7,8,9,3,3},{7,8,9,3,3},{7,8,9,3,3}}
                },{
                {{1,2,3,3,3},{1,2,3,3,3},{1,2,3,3,9}},
                {{4,5,6,3,3},{4,5,6,3,3},{4,5,6,3,3}},
                },{
                {{7,8,9,3,3},{7,8,9,3,3},{7,8,9,3,3}},
                {{7,8,9,3,3},{7,8,9,3,3},{7,8,9,3,3}}
            }});

            var pool = keras.layers.MaxPooling2D(pool_size: 2, strides: 1);
            var y = pool.Apply(x);

            Assert.AreEqual(4, y.shape[0]);
            Assert.AreEqual(1, y.shape[1]);
            Assert.AreEqual(2, y.shape[2]);
            Assert.AreEqual(5, y.shape[3]);

            var expected = np.array(new float[,,,]
            {
                {{{4.0f, 5.0f, 6.0f, 3.0f, 3.0f},
                 {4.0f, 5.0f, 6.0f, 9.0f, 3.0f}}},


               {{{7.0f, 8.0f, 9.0f, 3.0f, 3.0f},
                 {7.0f, 8.0f, 9.0f, 3.0f, 3.0f}}},


               {{{4.0f, 5.0f, 6.0f, 3.0f, 3.0f},
                 {4.0f, 5.0f, 6.0f, 3.0f, 9.0f}}},


               {{{7.0f, 8.0f, 9.0f, 3.0f, 3.0f},
                 {7.0f, 8.0f, 9.0f, 3.0f, 3.0f}}}
            });

            Assert.AreEqual(expected, y[0].numpy());
        }
    }
}
