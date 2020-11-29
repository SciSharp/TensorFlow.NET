using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using Tensorflow;
using static Tensorflow.KerasApi;

namespace TensorFlowNET.UnitTest.Keras
{
    [TestClass]
    public class LayersReshapingTest : EagerModeTestBase
    {
        [TestMethod]
        public void ZeroPadding2D()
        {
            var input_shape = new[] { 1, 1, 2, 2 };
            var x = np.arange(np.prod(input_shape)).reshape(input_shape);
            var zero_padding_2d = keras.layers.ZeroPadding2D(new[,] { { 1, 0 }, { 1, 0 } });
            var y = zero_padding_2d.Apply(x);
            Assert.AreEqual((1, 2, 3, 2), y.shape);
        }

        [TestMethod]
        public void UpSampling2D()
        {
            var input_shape = new[] { 2, 2, 1, 3 };
            var x = np.arange(np.prod(input_shape)).reshape(input_shape);
            var y = keras.layers.UpSampling2D(size: (1, 2)).Apply(x);
            Assert.AreEqual((2, 2, 2, 3), y.shape);
        }
    }
}
