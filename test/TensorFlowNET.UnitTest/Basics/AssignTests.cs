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
    }
}