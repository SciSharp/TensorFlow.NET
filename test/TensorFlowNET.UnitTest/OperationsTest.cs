using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow;

namespace TensorFlowNET.UnitTest
{
    [TestClass]
    public class OperationsTest
    {
        [TestMethod]
        public void constant()
        {
            tf.constant(4.0);
        }

        [TestMethod]
        public void placeholder()
        {
            var x = tf.placeholder(tf.float32);
        }

        [TestMethod]
        public void add()
        {
            var a = tf.placeholder(tf.float32);
            var b = tf.placeholder(tf.float32);
            var c = tf.add(a, b);

            //sess.run(adder_node, { a: 3, b: 4.5})
            //sess.run(adder_node, {a: [1,3], b: [2, 4]})
        }
    }
}
