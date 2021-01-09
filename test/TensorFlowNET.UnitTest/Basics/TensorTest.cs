using FluentAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using System;
using System.Linq;
using System.Runtime.InteropServices;
using Tensorflow;
using static Tensorflow.Binding;

namespace TensorFlowNET.UnitTest
{
    [TestClass, Ignore]
    public class TensorTest
    {
        [TestMethod]
        public void sparse_to_dense()
        {
            var indices = tf.reshape(tf.range(0, 5), new int[] { 5, 1 });
            var labels = tf.expand_dims(tf.constant(new[] { 0, 1, 2, 3, 4 }), 1);
            var st = tf.concat(values: new[] { indices, labels }, axis: 1);
            var onehot = tf.sparse_to_dense(st, (5, 5), 1);
            using (var sess = tf.Session())
            {
                var result = sess.run(onehot);
                Assert.IsTrue(Enumerable.SequenceEqual(new int[] { 1, 0, 0, 0, 0 }, result[0].ToArray<int>()));
                Assert.IsTrue(Enumerable.SequenceEqual(new int[] { 0, 1, 0, 0, 0 }, result[1].ToArray<int>()));
                Assert.IsTrue(Enumerable.SequenceEqual(new int[] { 0, 0, 1, 0, 0 }, result[2].ToArray<int>()));
                Assert.IsTrue(Enumerable.SequenceEqual(new int[] { 0, 0, 0, 1, 0 }, result[3].ToArray<int>()));
                Assert.IsTrue(Enumerable.SequenceEqual(new int[] { 0, 0, 0, 0, 1 }, result[4].ToArray<int>()));
            };
        }

        [TestMethod]
        public void sparse_tensor_to_dense()
        {
            var decoded_list = tf.SparseTensor(new[,]
            {
                { 0L, 0L },
                { 1L, 2L }
            },
            new int[] { 1, 2 },
            new[] { 3L, 4L });

            var onehot = tf.sparse_tensor_to_dense(decoded_list);
            using (var sess = tf.Session())
            {
                var result = sess.run(onehot);
                Assert.IsTrue(Enumerable.SequenceEqual(new int[] { 1, 0, 0, 0 }, result[0].ToArray<int>()));
                Assert.IsTrue(Enumerable.SequenceEqual(new int[] { 0, 0, 2, 0 }, result[1].ToArray<int>()));
                Assert.IsTrue(Enumerable.SequenceEqual(new int[] { 0, 0, 0, 0 }, result[2].ToArray<int>()));
            }
        }

        [TestMethod]
        public void batch_to_space_nd()
        {
            var inputs = np.arange(24).reshape(4, 2, 3);
            var block_shape = new[] { 2, 2 };
            int[,] crops = { { 0, 0 }, { 0, 0 } };
            var tensor = tf.batch_to_space_nd(inputs, block_shape, crops);

            using (var sess = tf.Session())
            {
                var result = sess.run(tensor);
                Assert.IsTrue(Enumerable.SequenceEqual(new int[] { 0, 6, 1, 7, 2, 8 }, result[0, 0].ToArray<int>()));
                Assert.IsTrue(Enumerable.SequenceEqual(new int[] { 12, 18, 13, 19, 14, 20 }, result[0, 1].ToArray<int>()));
                Assert.IsTrue(Enumerable.SequenceEqual(new int[] { 3, 9, 4, 10, 5, 11 }, result[0, 2].ToArray<int>()));
                Assert.IsTrue(Enumerable.SequenceEqual(new int[] { 15, 21, 16, 22, 17, 23 }, result[0, 3].ToArray<int>()));
            }
        }

        [TestMethod, Ignore]
        public void boolean_mask()
        {
            var tensor = new[] { 0, 1, 2, 3 };
            var mask = np.array(new[] { true, false, true, false });
            var masked = tf.boolean_mask(tensor, mask);
            using (var sess = tf.Session())
            {
                var result = sess.run(masked);
                Assert.IsTrue(Enumerable.SequenceEqual(new int[] { 0, 2 }, masked.ToArray<int>()));
            }
        }
    }
}