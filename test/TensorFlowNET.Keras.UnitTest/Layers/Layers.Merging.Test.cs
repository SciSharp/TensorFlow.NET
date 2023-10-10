using Microsoft.VisualStudio.TestTools.UnitTesting;
using System.Collections.Generic;
using Tensorflow.NumPy;
using static Tensorflow.KerasApi;

namespace Tensorflow.Keras.UnitTest.Layers
{
    [TestClass]
    public class LayersMergingTest : EagerModeTestBase
    {
        [TestMethod]
        [DataRow(1, 4, 1, 5)]
        [DataRow(2, 2, 2, 5)]
        [DataRow(3, 2, 1, 10)]
        public void Concatenate(int axis, int shapeA, int shapeB, int shapeC)
        {
            var x = np.arange(10).reshape((1, 2, 1, 5));
            var y = np.arange(10, 20).reshape((1, 2, 1, 5));
            var z = keras.layers.Concatenate(axis: axis).Apply(new Tensors(x, y));
            Assert.AreEqual((1, shapeA, shapeB, shapeC), z.shape);
        }

    }
}
