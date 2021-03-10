using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Utilities;
using Tensorflow;
using static Tensorflow.Binding;

namespace TensorFlowNET.UnitTest.ManagedAPI
{
    [TestClass]
    public class ArrayOpsTest : EagerModeTestBase
    {
        /// <summary>
        /// https://www.tensorflow.org/api_docs/python/tf/slice
        /// </summary>
        [TestMethod]
        public void Slice()
        {
            // Tests based on example code in TF documentation
            var input_array = tf.constant(np.array(new int[] { 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6 }).reshape(3,2,3));
            var indices = tf.constant(np.array(new int[] { 0, 2 }));

            var r1 = array_ops.slice(input_array, new int[] { 1, 0, 0 }, new int[] { 1, 1, 3 });
            Assert.AreEqual(new TensorShape(1,1,3), r1.shape);
            var r1np = r1.numpy();
            Assert.AreEqual(r1np[0, 0, 0], 3);
            Assert.AreEqual(r1np[0, 0, 1], 3);
            Assert.AreEqual(r1np[0, 0, 2], 3);


            var r2 = array_ops.slice(input_array, new int[] { 1, 0, 0 }, new int[] { 1, 2, 3 });
            Assert.AreEqual(new TensorShape(1, 2, 3), r2.shape);
            var r2np = r2.numpy();
            Assert.AreEqual(r2np[0, 0, 0], 3);
            Assert.AreEqual(r2np[0, 0, 1], 3);
            Assert.AreEqual(r2np[0, 0, 2], 3);
            Assert.AreEqual(r2np[0, 1, 0], 4);
            Assert.AreEqual(r2np[0, 1, 1], 4);
            Assert.AreEqual(r2np[0, 1, 2], 4);

            var r3 = array_ops.slice(input_array, new int[] { 1, 0, 0 }, new int[] { 2, 1, 3 });
            Assert.AreEqual(new TensorShape(2, 1, 3), r3.shape);
            var r3np = r3.numpy();
            Assert.AreEqual(r3np[0, 0, 0], 3);
            Assert.AreEqual(r3np[0, 0, 1], 3);
            Assert.AreEqual(r3np[0, 0, 2], 3);
            Assert.AreEqual(r3np[1, 0, 0], 5);
            Assert.AreEqual(r3np[1, 0, 1], 5);
            Assert.AreEqual(r3np[1, 0, 2], 5);
        }

        /// <summary>
        /// https://www.tensorflow.org/api_docs/python/tf/gather
        /// </summary>
        [TestMethod]
        public void Gather()
        {
            var input_array = tf.constant(np.arange(12).reshape(3, 4).astype(np.float32));
            var indices = tf.constant(np.array(new int[] { 0, 2 }));

            var result = array_ops.gather(input_array, indices);
            Assert.AreEqual(new TensorShape(2, 4), result.shape);
            Assert.AreEqual(result.numpy()[0, 0], 0.0f);
            Assert.AreEqual(result.numpy()[0, 1], 1.0f);
            Assert.AreEqual(result.numpy()[1, 3], 11.0f);

            // Tests based on example code in Python doc string for tf.gather()

            var p1 = tf.random.normal(new TensorShape(5, 6, 7, 8));
            var i1 = tf.random_uniform(new TensorShape(10, 11), maxval: 7, dtype: tf.int32);
            var r1 = tf.gather(p1, i1, axis:2);
            Assert.AreEqual(new TensorShape(5, 6, 10, 11, 8), r1.shape);

            var p2 = tf.random.normal(new TensorShape(4,3));
            var i2 = tf.constant(new int[,] { { 0, 2} });
            var r2 = tf.gather(p2, i2, axis: 0);
            Assert.AreEqual(new TensorShape(1, 2, 3), r2.shape);

            var r3 = tf.gather(p2, i2, axis: 1);
            Assert.AreEqual(new TensorShape(4,1,2), r3.shape);
        }
    }
}
