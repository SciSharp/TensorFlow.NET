using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Tensorflow;
using static Tensorflow.Binding;

namespace TensorFlowNET.UnitTest.TF_API
{
    [TestClass]
    public class MathApiTest : TFNetApiTest
    {
        // A constant vector of size 6
        Tensor a = tf.constant(new float[] { 1.0f, -0.5f, 3.4f, -2.1f, 0.0f, -6.5f });

        [TestMethod]
        public void Sin()
        {
            var b = tf.sin(a, name: "Sin");
            var expected = new float[] { 0.84147096f, -0.47942555f, -0.2555412f, -0.86320937f, 0f, -0.21511999f };
            var actual = b.ToArray<float>();
            Assert.IsTrue(Equal(expected, actual));
        }

        [TestMethod]
        public void Tan()
        {
            var b = tf.tan(a, name: "Tan");
            var expected = new float[] { 1.5574077f, -0.5463025f, 0.264317f, 1.709847f, 0f, -0.2202772f };
            var actual = b.ToArray<float>();
            Assert.IsTrue(Equal(expected, actual));
        }
    }
}
