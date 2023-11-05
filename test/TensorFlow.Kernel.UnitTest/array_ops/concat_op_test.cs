using Microsoft.VisualStudio.TestTools.UnitTesting;
using Tensorflow;
using Tensorflow.NumPy;
using static Tensorflow.Binding;

namespace TensorFlow.Kernel.UnitTest
{
    [TestClass]
    public class concat_op_test
    {
        [TestMethod]
        public void testConcatEmpty()
        {
            var t1 = tf.constant(new int[] { });
            var t2 = tf.constant(new int[] { });
            var c = array_ops.concat(new[] { t1, t2 }, 0);
            var expected = np.array(new int[] { });
            Assert.IsTrue(Enumerable.SequenceEqual(expected.ToArray<int>(), c.numpy().ToArray<int>()));
        }

        [TestMethod]
        public void testConcatNegativeAxis()
        {
            var t1 = tf.constant(new int[,] { { 1, 2, 3 }, { 4, 5, 6 } });
            var t2 = tf.constant(new int[,] { { 7, 8, 9 }, { 10, 11, 12 } });
            var c = array_ops.concat(new[] { t1, t2 }, -2);
            var expected = np.array(new int[,,] { { { 1, 2, 3 }, { 4, 5, 6 } }, { { 7, 8, 9 }, { 10, 11, 12 } } });
            Assert.IsTrue(Enumerable.SequenceEqual(expected.ToArray<int>(), c.numpy().ToArray<int>()));

            c = array_ops.concat(new[] { t1, t2 }, -1);
            expected = np.array(new int[,] { { 1, 2, 3, 7, 8, 9 }, { 4, 5, 6, 10, 11, 12 } });
            Assert.IsTrue(Enumerable.SequenceEqual(expected.ToArray<int>(), c.numpy().ToArray<int>()));
        }

        [TestMethod]
        [DataRow(TF_DataType.TF_INT32)]
        [DataRow(TF_DataType.TF_INT64)]
        [DataRow(TF_DataType.TF_UINT32)]
        [DataRow(TF_DataType.TF_UINT64)]
        public void testConcatDtype(TF_DataType dtype)
        {
            var t1 = tf.constant(new int[,] { { 1, 2, 3 }, { 4, 5, 6 } }, dtype: dtype);
            var t2 = tf.constant(new int[,] { { 7, 8, 9 }, { 10, 11, 12 } }, dtype: dtype);
            var c = array_ops.concat(new[] { t1, t2 }, 1);
            var expected = np.array(new int[,] { { 1, 2, 3, 7, 8, 9 }, { 4, 5, 6, 10, 11, 12 } });
            Assert.IsTrue(Enumerable.SequenceEqual(expected.ToArray<int>(), tf.cast(c, TF_DataType.TF_INT32).numpy().ToArray<int>()));

        }

        [TestMethod]
        [DataRow(TF_DataType.TF_INT32)]
        [DataRow(TF_DataType.TF_INT64)]
        public void testConcatAxisType(TF_DataType dtype)
        {
            var t1 = tf.constant(new int[,] { { 1, 2, 3 }, { 4, 5, 6 } });
            var t2 = tf.constant(new int[,] { { 7, 8, 9 }, { 10, 11, 12 } });
            var c = array_ops.concat(new[] { t1, t2 }, tf.constant(1, dtype: dtype));
            var expected = np.array(new int[,] { { 1, 2, 3, 7, 8, 9 }, { 4, 5, 6, 10, 11, 12 } });
            Assert.IsTrue(Enumerable.SequenceEqual(expected.ToArray<int>(), tf.cast(c, TF_DataType.TF_INT32).numpy().ToArray<int>()));
        }

    }
}