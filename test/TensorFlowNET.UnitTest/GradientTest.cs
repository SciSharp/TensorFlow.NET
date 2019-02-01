using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow;

namespace TensorFlowNET.UnitTest
{
    [TestClass]
    public class GradientTest
    {
        [TestMethod]
        public void Gradients()
        {
            var a = tf.constant(0.0);
            var b = 2.0 * a;
            var c = a + b;
            var g = tf.gradients(c, new Tensor[] { a, b }, stop_gradients: new Tensor[] { a, b });
        }
    }
}
