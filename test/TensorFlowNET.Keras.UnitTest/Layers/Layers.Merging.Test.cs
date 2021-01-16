using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using Tensorflow;
using static Tensorflow.KerasApi;

namespace TensorFlowNET.Keras.UnitTest
{
    [TestClass]
    public class LayersMergingTest : EagerModeTestBase
    {
        [TestMethod]
        public void Concatenate()
        {
            var x = np.arange(20).reshape(2, 2, 5);
            var y = np.arange(20, 30).reshape(2, 1, 5);
            var z = keras.layers.Concatenate(axis: 1).Apply(new Tensors(x, y));
            Assert.AreEqual((2, 3, 5), z.shape);
        }
    }
}
