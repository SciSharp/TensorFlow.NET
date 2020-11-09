using Microsoft.VisualStudio.TestTools.UnitTesting;
using static Tensorflow.Binding;

namespace TensorFlowNET.UnitTest.Basics
{
    [TestClass]
    public class VersionTest
    {
        [TestMethod]
        public void GetVersion()
        {
            var ver = tf.VERSION;
            Assert.IsTrue(ver.StartsWith("2."));
        }
    }
}
