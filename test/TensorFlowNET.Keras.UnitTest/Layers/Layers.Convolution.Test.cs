using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using Tensorflow;
using static Tensorflow.KerasApi;

namespace TensorFlowNET.Keras.UnitTest
{
    [TestClass]
    public class LayersConvolutionTest : EagerModeTestBase
    {
        [TestMethod]
        public void BasicConv2D()
        {
            var filters = 8;
            var conv = keras.layers.Conv2D(filters, activation: "linear");

            var x = np.arange(256.0f).reshape(1,8,8,4);
            var y = conv.Apply(x);

            Assert.AreEqual(4, y.shape.ndim);
            Assert.AreEqual(x.shape[0], y.shape[0]);
            Assert.AreEqual(x.shape[1] - 4, y.shape[1]);
            Assert.AreEqual(x.shape[2] - 4, y.shape[2]);
            Assert.AreEqual(filters, y.shape[3]);
        }

        [TestMethod]
        public void BasicConv2D_ksize()
        {
            var filters = 8;
            var conv = keras.layers.Conv2D(filters, kernel_size: 3, activation: "linear");

            var x = np.arange(256.0f).reshape(1, 8, 8, 4);
            var y = conv.Apply(x);

            Assert.AreEqual(4, y.shape.ndim);
            Assert.AreEqual(x.shape[0], y.shape[0]);
            Assert.AreEqual(x.shape[1] - 2, y.shape[1]);
            Assert.AreEqual(x.shape[2] - 2, y.shape[2]);
            Assert.AreEqual(filters, y.shape[3]);
        }

        [TestMethod]
        public void BasicConv2D_ksize_same()
        {
            var filters = 8;
            var conv = keras.layers.Conv2D(filters, kernel_size: 3, padding: "same", activation: "linear");

            var x = np.arange(256.0f).reshape(1, 8, 8, 4);
            var y = conv.Apply(x);

            Assert.AreEqual(4, y.shape.ndim);
            Assert.AreEqual(x.shape[0], y.shape[0]);
            Assert.AreEqual(x.shape[1], y.shape[1]);
            Assert.AreEqual(x.shape[2], y.shape[2]);
            Assert.AreEqual(filters, y.shape[3]);
        }

        [TestMethod]
        public void BasicConv2D_ksize_strides()
        {
            var filters = 8;
            var conv = keras.layers.Conv2D(filters, kernel_size: 3, strides: 2, activation: "linear");

            var x = np.arange(256.0f).reshape(1, 8, 8, 4);
            var y = conv.Apply(x);

            Assert.AreEqual(4, y.shape.ndim);
            Assert.AreEqual(x.shape[0], y.shape[0]);
            Assert.AreEqual(x.shape[1] - 5, y.shape[1]);
            Assert.AreEqual(x.shape[2] - 5, y.shape[2]);
            Assert.AreEqual(filters, y.shape[3]);
        }

        [TestMethod]
        public void BasicConv2D_ksize_dilation()
        {
            var filters = 8;
            var conv = keras.layers.Conv2D(filters, kernel_size: 3, dilation_rate: 2, activation: "linear");

            var x = np.arange(256.0f).reshape(1, 8, 8, 4);
            var y = conv.Apply(x);

            Assert.AreEqual(4, y.shape.ndim);
            Assert.AreEqual(x.shape[0], y.shape[0]);
            Assert.AreEqual(x.shape[1] - 4, y.shape[1]);
            Assert.AreEqual(x.shape[2] - 4, y.shape[2]);
            Assert.AreEqual(filters, y.shape[3]);
        }

        [TestMethod]
        public void BasicConv2D_ksize_dilation_same()
        {
            var filters = 8;
            var conv = keras.layers.Conv2D(filters, kernel_size: 3, dilation_rate: 2, padding: "same", activation: "linear");

            var x = np.arange(256.0f).reshape(1, 8, 8, 4);
            var y = conv.Apply(x);

            Assert.AreEqual(4, y.shape.ndim);
            Assert.AreEqual(x.shape[0], y.shape[0]);
            Assert.AreEqual(x.shape[1], y.shape[1]);
            Assert.AreEqual(x.shape[2], y.shape[2]);
            Assert.AreEqual(filters, y.shape[3]);
        }
    }
}
