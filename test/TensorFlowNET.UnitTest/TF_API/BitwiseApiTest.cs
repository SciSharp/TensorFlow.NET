using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Tensorflow;
using static Tensorflow.Binding;

namespace TensorFlowNET.UnitTest.TF_API
{
    [TestClass]
    public class BitwiseApiTest : TFNetApiTest
    {
        Tensor lhs = tf.constant(new int[] { -1, -5, -3, -14 });
        Tensor rhs = tf.constant(new int[] { 5, 0, 7, 11 });

        [TestMethod]
        public void LeftShift()
        {
            var left_shift_result = tf.bitwise.left_shift(lhs, rhs);
            var expected = new int[] { -32, -5, -384, -28672 };
            var actual = left_shift_result.ToArray<int>();
            Assert.IsTrue(Enumerable.SequenceEqual(expected, actual));
        }
    }
}
