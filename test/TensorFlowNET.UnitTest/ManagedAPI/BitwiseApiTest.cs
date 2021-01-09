using Microsoft.VisualStudio.TestTools.UnitTesting;
using System.Linq;
using Tensorflow;
using static Tensorflow.Binding;

namespace TensorFlowNET.UnitTest.ManagedAPI
{
    [TestClass]
    public class BitwiseApiTest : EagerModeTestBase
    {
        [TestInitialize]
        public void Init()
        {
            tf.enable_eager_execution();
        }

        [TestMethod]
        public void BitwiseAnd()
        {
            Tensor lhs = tf.constant(new int[] { 0, 5, 3, 14 });
            Tensor rhs = tf.constant(new int[] { 5, 0, 7, 11 });

            var bitwise_and_result = tf.bitwise.bitwise_and(lhs, rhs);
            var expected = new int[] { 0, 0, 3, 10 };
            var actual = bitwise_and_result.ToArray<int>();
            Assert.IsTrue(Enumerable.SequenceEqual(expected, actual));
        }

        [TestMethod]
        public void BitwiseOr()
        {
            Tensor lhs = tf.constant(new int[] { 0, 5, 3, 14 });
            Tensor rhs = tf.constant(new int[] { 5, 0, 7, 11 });

            var bitwise_or_result = tf.bitwise.bitwise_or(lhs, rhs);
            var expected = new int[] { 5, 5, 7, 15 };
            var actual = bitwise_or_result.ToArray<int>();
            Assert.IsTrue(Enumerable.SequenceEqual(expected, actual));
        }

        [TestMethod]
        public void BitwiseXOR()
        {
            Tensor lhs = tf.constant(new int[] { 0, 5, 3, 14 });
            Tensor rhs = tf.constant(new int[] { 5, 0, 7, 11 });

            var bitwise_xor_result = tf.bitwise.bitwise_xor(lhs, rhs);
            var expected = new int[] { 5, 5, 4, 5 };
            var actual = bitwise_xor_result.ToArray<int>();
            Assert.IsTrue(Enumerable.SequenceEqual(expected, actual));
        }

        [TestMethod]
        public void Invert()
        {
            Tensor lhs = tf.constant(new int[] { 0, 1, -3, int.MaxValue });

            var invert_result = tf.bitwise.invert(lhs);
            var expected = new int[] { -1, -2, 2, int.MinValue };
            var actual = invert_result.ToArray<int>();
            Assert.IsTrue(Enumerable.SequenceEqual(expected, actual));
        }

        [TestMethod]
        public void LeftShift()
        {
            Tensor lhs = tf.constant(new int[] { -1, -5, -3, -14 });
            Tensor rhs = tf.constant(new int[] { 5, 0, 7, 11 });

            var left_shift_result = tf.bitwise.left_shift(lhs, rhs);
            var expected = new int[] { -32, -5, -384, -28672 };
            var actual = left_shift_result.ToArray<int>();
            Assert.IsTrue(Enumerable.SequenceEqual(expected, actual));
        }

        [TestMethod]
        public void RightShift()
        {
            Tensor lhs = tf.constant(new int[] { -2, 64, 101, 32 });
            Tensor rhs = tf.constant(new int[] { -1, -5, -3, -14 });

            var right_shift_result = tf.bitwise.right_shift(lhs, rhs);
            var expected = new int[] { -2, 64, 101, 32 };
            var actual = right_shift_result.ToArray<int>();
            Assert.IsTrue(Enumerable.SequenceEqual(expected, actual));
        }

    }
}
