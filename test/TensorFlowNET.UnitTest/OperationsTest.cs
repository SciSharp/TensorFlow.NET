using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Core;
using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow;
using Buffer = Tensorflow.Buffer;

namespace TensorFlowNET.UnitTest
{
    [TestClass]
    public class OperationsTest
    {
        /// <summary>
        /// Port from tensorflow\c\c_api_test.cc
        /// `TEST(CAPI, GetAllOpList)`
        /// </summary>
        [TestMethod]
        public void GetAllOpList()
        {
            var handle = c_api.TF_GetAllOpList();
            var buffer = new Buffer(handle);
            Assert.IsTrue(buffer.Length == buffer.Length);
        }

        [TestMethod]
        public void addInPlaceholder()
        {
            var a = tf.placeholder(tf.float32);
            var b = tf.placeholder(tf.float32);
            var c = tf.add(a, b);

            using(var sess = tf.Session())
            {
                var o = sess.run(c, feed_dict: new FeedItem[]
                {
                    new FeedItem(a, 3.0f),
                    new FeedItem(b, 2.0f)
                });
                Assert.AreEqual((float)o, 5.0f);
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
                Assert.AreEqual((float)o, 9.0f);
            }
        }
    }
}
