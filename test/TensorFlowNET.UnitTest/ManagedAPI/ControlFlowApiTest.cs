using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using Tensorflow;
using static Tensorflow.Binding;

namespace TensorFlowNET.UnitTest.ManagedAPI
{
    [TestClass]
    public class ControlFlowApiTest
    {
        [TestMethod]
        public void WhileLoopOneInputEagerMode()
        {
            tf.enable_eager_execution();

            var i = tf.constant(2);
            Func<Tensor, Tensor> c = (x) => tf.less(x, 10);
            Func<Tensor, Tensor> b = (x) => tf.add(x, 1);
            var r = tf.while_loop(c, b, i);
            Assert.AreEqual(10, (int)r);
        }

        [TestMethod]
        public void WhileLoopTwoInputsEagerMode()
        {
            tf.enable_eager_execution();

            var i = tf.constant(2);
            var j = tf.constant(3);
            Func<Tensor[], Tensor> c = (x) => tf.less(x[0] + x[1], 10);
            Func<Tensor[], Tensor[]> b = (x) => new[] { tf.add(x[0], 1), tf.add(x[1], 1) };
            var r = tf.while_loop(c, b, new[] { i, j });
            Assert.AreEqual(5, (int)r[0]);
            Assert.AreEqual(6, (int)r[1]);
        }

        [TestMethod, Ignore]
        public void WhileLoopGraphMode()
        {
            tf.compat.v1.disable_eager_execution();

            var i = tf.constant(2);
            Func<Tensor, Tensor> c = (x) => tf.less(x, 10);
            Func<Tensor, Tensor> b = (x) => tf.add(x, 1);
            var r = tf.while_loop(c, b, i);
            Assert.AreEqual(10, (int)r);
        }
    }
}
