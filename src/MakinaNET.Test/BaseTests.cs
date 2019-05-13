using System;
using Tensorflow;
using Makina;
using Makina.Layers;
using NumSharp;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace Makina.Test
{
    [TestClass]
    public class BaseTests
    {
        [TestMethod]
        public void Dense_Tensor_ShapeTest()
        {
            var dense_1 = new Dense(1, name: "dense_1", activation: tf.nn.relu());
            var input = new Tensor(np.array(new int[] { 3 }));
            dense_1.__build__(input.getShape());
            var outputShape = dense_1.output_shape(input.getShape());
            //Assert.AreEqual(outputShape.Dimensions, new int[] { 1 });
        }
    }
}
