using Microsoft.VisualStudio.TestTools.UnitTesting;
using static Tensorflow.Binding;
using Tensorflow.NumPy;

namespace TensorFlowNET.UnitTest.NenuralNetwork
{
    [TestClass]
    public class NeuralNetworkTest : EagerModeTestBase
    {
        [TestMethod]
        public void l2_loss()
        {
            var x = tf.Variable(np.array(new[,] { { 1, 2, 3, 4 }, { 5, 6, 7, 8 } }), dtype: tf.float32);
            var l2 = tf.nn.l2_loss(x);
            Assert.AreEqual(l2.numpy(), 102f);
        }
    }
}
