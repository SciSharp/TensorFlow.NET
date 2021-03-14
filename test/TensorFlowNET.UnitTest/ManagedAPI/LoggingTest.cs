using Microsoft.VisualStudio.TestTools.UnitTesting;
using static Tensorflow.Binding;

namespace TensorFlowNET.UnitTest.ManagedAPI
{
    [TestClass]
    public class LoggingTest
    {
        [TestMethod]
        public void PrintTest()
        {
            var tensor = tf.range(10);
            tf.print(tensor);
        }
    }
}
