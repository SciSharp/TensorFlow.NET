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

            using(var sess = tf.Session())
            {
                var feed_dict = new Dictionary<Tensor, object>();
                feed_dict.Add(a, 3);
                feed_dict.Add(b, 2);

                sess.run(c, feed_dict);
            }
        }
    }
}
