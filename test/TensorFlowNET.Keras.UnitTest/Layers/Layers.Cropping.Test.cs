using Microsoft.VisualStudio.TestTools.UnitTesting;
using Tensorflow.NumPy;
using static Tensorflow.Binding;
using static Tensorflow.KerasApi;

namespace Tensorflow.Keras.UnitTest.Layers
{
    [TestClass]
    public class LayersCroppingTest : EagerModeTestBase
    {
        [TestMethod]
        public void Cropping1D()
        {
            Shape input_shape = (1, 5, 2);
            var x = tf.zeros(input_shape);
            var cropping_1d = keras.layers.Cropping1D(new[] { 1, 2 });
            var y = cropping_1d.Apply(x);
            Assert.AreEqual((1, 2, 2), y.shape);
        }

        [TestMethod]
        public void Cropping2D()
        {
            Shape input_shape = (1, 5, 6, 1);
            NDArray cropping = new NDArray(new[,] { { 1, 2 }, { 1, 3 } });
            var x = tf.zeros(input_shape);
            var cropping_2d = keras.layers.Cropping2D(cropping);
            var y = cropping_2d.Apply(x);
            Assert.AreEqual((1, 2, 2, 1), y.shape);
        }

        [TestMethod]
        public void Cropping3D()
        {
            Shape input_shape = new Shape(1, 5, 6, 7, 1);
            NDArray cropping = new NDArray(new[,] { { 1, 2 }, { 1, 3 }, { 1, 4 } });
            var x = tf.zeros(input_shape);
            var cropping_3d = keras.layers.Cropping3D(cropping);
            var y = cropping_3d.Apply(x);
            Assert.AreEqual(new Shape(1, 2, 2, 2, 1), y.shape);
        }
    }
}
