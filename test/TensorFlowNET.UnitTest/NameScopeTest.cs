using Microsoft.VisualStudio.TestTools.UnitTesting;
using Tensorflow;
using static Tensorflow.Binding;

namespace TensorFlowNET.UnitTest.Basics
{
    [TestClass]
    public class NameScopeTest : EagerModeTestBase
    {
        string name = "";

        [TestMethod]
        public void NameScopeInEagerMode()
        {
            tf_with(new ops.NameScope("scope"), scope =>
            {
                string name = scope;
                var const1 = tf.constant(1.0);
            });
        }
    }
}
