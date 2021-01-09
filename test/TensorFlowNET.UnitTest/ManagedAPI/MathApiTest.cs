using Microsoft.VisualStudio.TestTools.UnitTesting;
using System.Linq;
using Tensorflow;
using static Tensorflow.Binding;

namespace TensorFlowNET.UnitTest.ManagedAPI
{
    [TestClass]
    public class MathApiTest : EagerModeTestBase
    {
        // A constant vector of size 6
        Tensor a = tf.constant(new float[] { 1.0f, -0.5f, 3.4f, -2.1f, 0.0f, -6.5f });
        Tensor b = tf.constant(new float[,] { { 1.0f, -0.5f, 3.4f }, { -2.1f, 0.0f, -6.5f } });

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

        [TestMethod]
        public void ReduceSum()
        {
            var x1 = tf.reduce_sum(b);
            Assert.AreEqual(-4.7f, (float)x1);

            var x2 = tf.reduce_sum(b, 0);
            Assert.IsTrue(Enumerable.SequenceEqual(new[] { -1.0999999f, -0.5f, -3.1f }, x2.ToArray<float>()));

            var x3 = tf.reduce_sum(b, 1);
            Assert.IsTrue(Enumerable.SequenceEqual(new[] { 3.9f, -8.6f }, x3.ToArray<float>()));

            var x4 = tf.reduce_sum(b, 1, keepdims: true);
            Assert.AreEqual((2, 1), x4.TensorShape);

            var x5 = tf.reduce_sum(b, (0, 1));
            Assert.AreEqual(-4.7f, (float)x5);
        }
    }
}
