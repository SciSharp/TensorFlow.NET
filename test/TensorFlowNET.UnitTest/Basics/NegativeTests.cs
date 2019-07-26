﻿using Microsoft.VisualStudio.TestTools.UnitTesting;
using Tensorflow;
using static Tensorflow.Python;

namespace TensorFlowNET.UnitTest.Basics
{
    [TestClass]
    public sealed class NegativeTests
    {
        [TestMethod]
        public void ShouldReturnNegative()
        {
            var x = tf.constant(new[,] { { 1, 2 } });
            var neg_x = tf.negative(x);
            with(tf.Session(), session =>
            {
                var result = session.run(neg_x);

                Assert.AreEqual(result[0][0], -1);
                Assert.AreEqual(result[0][1], -2);
            });
        }
    }
}
