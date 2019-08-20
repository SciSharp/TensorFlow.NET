using Microsoft.VisualStudio.TestTools.UnitTesting;
using Tensorflow;
using static Tensorflow.Binding;

namespace TensorFlowNET.UnitTest
{
    [TestClass]
    public class ConsumersTest : CApiTest
    {
        [TestMethod]
        public void Constant()
        {
            var X = tf.placeholder(tf.float64);
            var W = tf.constant(1.0D);

            var mul = tf.multiply(X, W);
            EXPECT_EQ(1, X.op.OutputNumConsumers(0));
            EXPECT_EQ(1, W.op.OutputNumConsumers(0));
        }

        [TestMethod]
        public void Variable()
        {
            var X = tf.placeholder(tf.float64);
            var W = tf.Variable(1.0D, name: "var");

            var mul = tf.multiply(X, W);
            EXPECT_EQ(1, X.op.OutputNumConsumers(0));
            //EXPECT_EQ(1, W.op.OutputNumConsumers(0));
        }
    }
}
