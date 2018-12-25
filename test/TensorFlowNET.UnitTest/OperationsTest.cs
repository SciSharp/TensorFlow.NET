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
            var x = tf.constant(4.0f);
        }

        [TestMethod]
        public void placeholder()
        {
            var x = tf.placeholder(tf.float32);
        }

        [TestMethod]
        public void addInPlaceholder()
        {
            var a = tf.placeholder(tf.float32);
            var b = tf.placeholder(tf.float32);
            var c = tf.add(a, b);

            using(var sess = tf.Session())
            {
                var feed_dict = new Dictionary<Tensor, object>();
                feed_dict.Add(a, 3.0f);
                feed_dict.Add(b, 2.0f);

                var o = sess.run(c, feed_dict);
            }
        }

        [TestMethod]
        public void addInConstant()
        {
            var a = tf.constant(4.0f);
            var b = tf.constant(5.0f);
            var c = tf.add(a, b);

            using (var sess = tf.Session())
            {
                var o = sess.run(c);
                Assert.AreEqual(o, 9.0f);
            }
        }
    }
}
