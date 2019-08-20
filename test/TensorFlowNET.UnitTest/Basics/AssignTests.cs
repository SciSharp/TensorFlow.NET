using Microsoft.VisualStudio.TestTools.UnitTesting;
using static Tensorflow.Python;
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

            spike.initializer.run();
            foreach (var i in range(1, 2))
            {
                if (raw_data[i] - raw_data[i - 1] > 5d)
                {
                    var updater = tf.assign(spike, tf.constant(true));
                    updater.eval();
                }
                else
                {
                    tf.assign(spike, tf.constant(true)).eval();
                }

                Assert.AreEqual((bool)spike.eval(), expected[i - 1]);
            }
        }
    }
}