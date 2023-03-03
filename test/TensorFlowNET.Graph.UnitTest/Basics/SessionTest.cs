using Microsoft.VisualStudio.TestTools.UnitTesting;
using Tensorflow.NumPy;
using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow;
using static Tensorflow.Binding;

namespace TensorFlowNET.UnitTest.Basics
{
    [TestClass]
    public class SessionTest : GraphModeTestBase
    {
        [TestMethod]
        public void EvalTensor()
        {
            lock (this)
            {
                var a = constant_op.constant(np.array(3.0).reshape((1, 1)));
                var b = constant_op.constant(np.array(2.0).reshape((1, 1)));
                var c = math_ops.matmul(a, b, name: "matmul");
                var sess = tf.Session();
                var result = c.eval(sess);
                Assert.AreEqual(result[0], 6.0);
            }
        }

        [TestMethod]
        public void Eval_SmallString_Scalar()
        {
            var a = constant_op.constant("123 heythere 123 ", TF_DataType.TF_STRING);
            var c = tf.strings.substr(a, 4, 8);
            var sess = tf.Session();
            var result = c.eval(sess).StringData();
            Assert.AreEqual(result[0], "heythere");
        }

        [TestMethod]
        public void Eval_LargeString_Scalar()
        {
            lock (this)
            {
                const int size = 30_000;
                var a = constant_op.constant(new string('a', size), TF_DataType.TF_STRING);
                var c = tf.strings.substr(a, 0, size - 5000);
                var sess = tf.Session();
                var result = UTF8Encoding.UTF8.GetString(c.eval(sess).ToByteArray());
                Console.WriteLine(result);
            }
        }

        [TestMethod]
        public void Autocast_Case0()
        {
            var sess = tf.Session().as_default();
            ITensorOrOperation operation = tf.global_variables_initializer();
            // the cast to ITensorOrOperation is essential for the test of this method signature
            var ret = sess.run(operation);
        }

        [TestMethod]
        public void Autocast_Case1()
        {
            var sess = tf.Session().as_default();
            var input = tf.placeholder(tf.int32, shape: new Shape(6));
            var op = tf.reshape(input, new int[] { 2, 3 });
            sess.run(tf.global_variables_initializer());
            var ret = sess.run(op, feed_dict: (input, np.array(1, 2, 3, 4, 5, 6)));

            Assert.AreEqual(ret.shape, (2, 3));
            assertAllEqual(ret.ToArray<int>(), new[] { 1, 2, 3, 4, 5, 6 });
            print(ret.dtype);
            print(ret);
        }

        [TestMethod]
        public void Autocast_Case2()
        {
            var sess = tf.Session().as_default();
            var input = tf.placeholder(tf.float32, shape: new Shape(6));
            var op = tf.reshape(input, new int[] { 2, 3 });
            sess.run(tf.global_variables_initializer());
            var ret = sess.run(op, feed_dict: (input, np.array(1, 2, 3, 4, 5, 6).astype(np.float32) + 0.1f));
        }

        [TestMethod, Ignore]
        public void Autocast_Case3()
        {
            var sess = tf.Session().as_default();
            var input = tf.placeholder(tf.float32, shape: new Shape(6));
            var op = tf.reshape(input, new int[] { 2, 3 });
            sess.run(tf.global_variables_initializer());
            var ret = sess.run(op, feed_dict: (input, np.array(1, 2, 3, 4, 5, 6).astype(np.float32) + 0.1f));

            Assert.AreEqual(ret.shape, (2, 3));
            Assert.AreEqual(ret, new[] { 1, 2, 3, 4, 5, 6 });
            print(ret.dtype);
            print(ret);
        }

        [TestMethod, Ignore]
        public void Autocast_Case4()
        {
            var sess = tf.Session().as_default();
            var input = tf.placeholder(tf.byte8, shape: new Shape(6));
            var op = tf.reshape(input, new int[] { 2, 3 });
            sess.run(tf.global_variables_initializer());
            var ret = sess.run(op, feed_dict: (input, np.array(1, 2, 3, 4, 5, 6).astype(np.float32) + 0.1f));

            Assert.AreEqual(ret.shape, (2, 3));
            Assert.AreEqual(ret, new[] { 1, 2, 3, 4, 5, 6 });
            print(ret.dtype);
            print(ret);
        }
    }
}