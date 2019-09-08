using System;
using FluentAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using Tensorflow;
using static Tensorflow.Binding;

namespace TensorFlowNET.UnitTest.layers_test
{
    [TestClass]
    public class flatten
    {
        [TestMethod]
        public void Case1()
        {
            var sess = tf.Session().as_default();

            var input = tf.placeholder(TF_DataType.TF_INT32, new TensorShape(3, 4, 3, 1, 2));
            sess.run(tf.layers.flatten(input), (input, np.arange(3 * 4 * 3 * 1 * 2).reshape(3, 4, 3, 1, 2))).Should().BeShaped(3, 24);
        }

        [TestMethod]
        public void Case2()
        {
            var sess = tf.Session().as_default();

            var input = tf.placeholder(TF_DataType.TF_INT32, new TensorShape(6));
            sess.run(tf.layers.flatten(input), (input, np.arange(6))).Should().BeShaped(6, 1);
        }

        [TestMethod]
        public void Case3()
        {
            var sess = tf.Session().as_default();

            var input = tf.placeholder(TF_DataType.TF_INT32, new TensorShape());
            new Action(() => sess.run(tf.layers.flatten(input), (input, NDArray.Scalar(6)))).Should().Throw<ValueError>();
        }

        [TestMethod]
        public void Case4()
        {
            var sess = tf.Session().as_default();

            var input = tf.placeholder(TF_DataType.TF_INT32, new TensorShape(3, 4, None, 1, 2));
            sess.run(tf.layers.flatten(input), (input, np.arange(3 * 4 * 3 * 1 * 2).reshape(3, 4, 3, 1, 2))).Should().BeShaped(3, 24);
        }

        [TestMethod]
        public void Case5()
        {
            var sess = tf.Session().as_default();

            var input = tf.placeholder(TF_DataType.TF_INT32, new TensorShape(None, 4, 3, 1, 2));
            sess.run(tf.layers.flatten(input), (input, np.arange(3 * 4 * 3 * 1 * 2).reshape(3, 4, 3, 1, 2))).Should().BeShaped(3, 24);
        }
    }
}