using Microsoft.VisualStudio.TestTools.UnitTesting;
using Tensorflow;
using static Tensorflow.Python;

namespace TensorFlowNET.UnitTest
{
    [TestClass]
    public class PlaceholderTest
    {
        [TestMethod]
        public void placeholder()
        {
            var x = tf.placeholder(tf.int32);
            var y = x * 3;

            using (var sess = tf.Session())
            {
                var result = sess.run(y, 
                    new FeedItem(x, 2));
                Assert.AreEqual((int)result[0], 6);
            }
        }
    }
}
