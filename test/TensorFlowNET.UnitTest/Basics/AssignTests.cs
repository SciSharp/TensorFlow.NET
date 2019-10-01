using Microsoft.VisualStudio.TestTools.UnitTesting;
using Tensorflow;
using static Tensorflow.Binding;

namespace TensorFlowNET.UnitTest.Basics
{
    [TestClass]
    public sealed class AssignTests
    {
        [Ignore("Not implemented")]
        [TestMethod]
        public void ShouldAssignVariable()
        {
            var raw_data = new[] { 1.0, 2.0, 8.0, -1.0, 0.0, 5.5, 6.0, 16.0 };
            var expected = new[] { false, true, false, false, true, false, true };

            var spike = tf.Variable(false);
            using (var sess = new Session())
            {
                spike.initializer.run(session: sess);
                foreach (var i in range(1, 2))
                {
                    if (raw_data[i] - raw_data[i - 1] > 5d)
                    {
                        var updater = tf.assign(spike, tf.constant(true));
                        updater.eval(sess);
                    } else
                    {
                        tf.assign(spike, tf.constant(true)).eval(sess);
                    }

                    Assert.AreEqual((bool) spike.eval(), expected[i - 1]);
                }
            }
        }

        [TestMethod]
        public void Bug397()
        {
            // fix bug https://github.com/SciSharp/TensorFlow.NET/issues/397
            var W = tf.Variable(-1, name: "weight_" + 1, dtype: tf.float32);
            var init = tf.global_variables_initializer();
            var reluEval = tf.nn.relu(W);
            var nonZero = tf.assign(W, reluEval);

            using (var sess = tf.Session())
            {
                sess.run(init);
                float result = nonZero.eval();
                Assert.IsTrue(result == 0f);
            }
        }
    }
}