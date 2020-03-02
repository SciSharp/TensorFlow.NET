using Microsoft.VisualStudio.TestTools.UnitTesting;
using Tensorflow;
using static Tensorflow.Binding;

namespace TensorFlowNET.UnitTest
{
    [TestClass]
    public class VersionTest
    {
        [TestMethod]
        public void GetVersion()
        {
            var ver = tf.VERSION;
            Assert.IsTrue(ver.StartsWith("1.15."));
        }
    }
}
