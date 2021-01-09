using Microsoft.VisualStudio.TestTools.UnitTesting;
using Tensorflow;
using static Tensorflow.Binding;

namespace TensorFlowNET.UnitTest.nn_test
{
    [TestClass]
    public class ActivationFunctionTest : EagerModeTestBase
    {
        // A constant vector of size 6
        Tensor a = tf.constant(new float[] { 1.0f, -0.5f, 3.4f, -2.1f, 0.0f, -6.5f });

        [TestMethod]
        public void Sigmoid()
        {
            var b = tf.nn.sigmoid(a, name: "sigmoid");
            var expected = new float[] { 0.7310586f, 0.37754068f, 0.9677046f, 0.10909683f, 0.5f, 0.00150118f };
            var actual = b.ToArray<float>();
            Assert.IsTrue(Equal(expected, actual));
        }

        [TestMethod]
        public void ReLU()
        {
            var b = tf.nn.relu(a, name: "ReLU");
            var expected = new float[] { 1f, 0f, 3.4f, 0f, 0f, 0f };
            var actual = b.ToArray<float>();
            Assert.IsTrue(Equal(expected, actual));
        }

        [TestMethod]
        public void TanH()
        {
            var b = tf.nn.tanh(a, name: "TanH");
            var expected = new float[] { 0.7615942f, -0.46211717f, 0.9977749f, -0.970452f, 0f, -0.99999547f };
            var actual = b.ToArray<float>();
            Assert.IsTrue(Equal(expected, actual));
        }
    }
}
