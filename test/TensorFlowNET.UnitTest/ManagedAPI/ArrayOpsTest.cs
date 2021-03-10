using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using Tensorflow;
using static Tensorflow.Binding;

namespace TensorFlowNET.UnitTest.ManagedAPI
{
    [TestClass]
    public class ArrayOpsTest : EagerModeTestBase
    { 
        /// <summary>
        /// https://www.tensorflow.org/api_docs/python/tf/keras/layers/Embedding
        /// </summary>
        [TestMethod]
        public void Gather()
        {
            var input_array = tf.constant(np.arange(12).reshape(3, 4).astype(np.float32));
            var indices = tf.constant(np.array(new int[] { 0, 2 }));

            var result = array_ops.gather(input_array, indices);
            Assert.AreEqual(new TensorShape(2, 4), result.shape);
            Assert.AreEqual(result.numpy()[0,0], 0.0f);
            Assert.AreEqual(result.numpy()[0,1], 1.0f);
            Assert.AreEqual(result.numpy()[1,3], 11.0f);
        }
    }
}
