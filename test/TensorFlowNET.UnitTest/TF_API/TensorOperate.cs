using FluentAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using System.Linq;
using Tensorflow;
using static Tensorflow.Binding;

namespace Tensorflow.UnitTest.TF_API
{
    [TestClass]
    public class TensorOperate
    {
        [TestMethod]
        public void TransposeTest()
        {
            var a = tf.constant(np.array(new[, , ,] { { { { 1, 11, 2, 22 } }, { { 3, 33, 4, 44 } } },
                { { { 5, 55, 6, 66 } }, { { 7, 77, 8, 88 } } } }));
            var b = tf.transpose(a, new[] { 3, 1, 2, 0 });
            var transpose_a = tf.constant(np.array(new[, , ,] { { { { 1, 5 } }, { { 3, 7 } } },
                { { { 11, 55 } }, { { 33, 77 } } }, { { { 2, 6 } }, { { 4, 8 } } },
                { { { 22, 66 } }, { { 44, 88 } } } }));
            Assert.IsTrue(Enumerable.SequenceEqual(new[] { 4, 2, 1, 2 }, b.shape));
            Assert.IsTrue(Enumerable.SequenceEqual(transpose_a.numpy().ToArray<int>(), b.numpy().ToArray<int>()));
        }

        [TestMethod]
        public void InitTensorTest()
        {
            var a = tf.constant(new NDArray(new[, ,] { { { 1 }, { 2 }, { 3 } }, { { 4 }, { 5 }, { 6 } } }));
            var b = tf.constant(new[, ,] { { { 1 }, { 2 }, { 3 } }, { { 4 }, { 5 }, { 6 } } });
            //Test Result : a is OK , and b is error .
            Assert.IsTrue(Enumerable.SequenceEqual(new[] { 2, 3, 1 }, a.shape));
            Assert.IsTrue(Enumerable.SequenceEqual(new[] { 2, 3, 1 }, b.shape));
        }
    }
}
