using FluentAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow;
using Tensorflow.Util;
using static Tensorflow.Binding;

namespace TensorFlowNET.UnitTest
{
    [TestClass, Ignore]
    public class SessionTest
    {
        [TestMethod]
        public void EvalTensor()
        {
            lock (this)
            {
                var a = constant_op.constant(np.array(3.0).reshape(1, 1));
                var b = constant_op.constant(np.array(2.0).reshape(1, 1));
                var c = math_ops.matmul(a, b, name: "matmul");
                using (var sess = tf.Session())
                {
                    var result = c.eval(sess);
                    Assert.AreEqual(6, result.GetAtIndex<double>(0));
                }
            }
        }

        [TestMethod]
        public void Eval_SmallString_Scalar()
        {
            lock (this)
            {
                var a = constant_op.constant("123 heythere 123 ", TF_DataType.TF_STRING);
                var c = tf.strings.substr(a, 4, 8);
                using (var sess = tf.Session())
                {
                    var result = UTF8Encoding.UTF8.GetString((byte[])c.eval(sess));
                    Console.WriteLine(result);
                    result.Should().Be("heythere");
                }
            }
        }

        [TestMethod]
        public void Eval_LargeString_Scalar()
        {
            lock (this)
            {
                const int size = 30_000;
                var a = constant_op.constant(new string('a', size), TF_DataType.TF_STRING);
                var c = tf.strings.substr(a, 0, size - 5000);
                using (var sess = tf.Session())
                {
                    var result = UTF8Encoding.UTF8.GetString((byte[])c.eval(sess));
                    Console.WriteLine(result);
                    result.Should().HaveLength(size - 5000).And.ContainAll("a");
                }
            }
        }

        [TestMethod]
        public void Autocast_Case0()
        {
            var sess = tf.Session().as_default();
            ITensorOrOperation operation = tf.global_variables_initializer();
            // the cast to ITensorOrOperation is essential for the test of this method signature
            var ret = sess.run(operation);

            ret.Should().BeNull();
        }

        [TestMethod]
        public void Autocast_Case1()
        {
            var sess = tf.Session().as_default();
            var input = tf.placeholder(tf.float32, shape: new TensorShape(6));
            var op = tf.reshape(input, new int[] { 2, 3 });
            sess.run(tf.global_variables_initializer());
            var ret = sess.run(op, feed_dict: (input, np.array(1, 2, 3, 4, 5, 6)));

            ret.Should().BeOfType<float>().And.BeShaped(2, 3).And.BeOfValues(1, 2, 3, 4, 5, 6);
            print(ret.dtype);
            print(ret);
        }

        [TestMethod]
        public void Autocast_Case2()
        {
            var sess = tf.Session().as_default();
            var input = tf.placeholder(tf.float64, shape: new TensorShape(6));
            var op = tf.reshape(input, new int[] { 2, 3 });
            sess.run(tf.global_variables_initializer());
            var ret = sess.run(op, feed_dict: (input, np.array(1, 2, 3, 4, 5, 6).astype(NPTypeCode.Single) + 0.1f));

            ret.Should().BeOfType<double>().And.BeShaped(2, 3).And.BeOfValuesApproximately(0.001d, 1.1, 2.1, 3.1, 4.1, 5.1, 6.1);
            print(ret.dtype);
            print(ret);
        }

        [TestMethod]
        public void Autocast_Case3()
        {
            var sess = tf.Session().as_default();
            var input = tf.placeholder(tf.int64, shape: new TensorShape(6));
            var op = tf.reshape(input, new int[] { 2, 3 });
            sess.run(tf.global_variables_initializer());
            var ret = sess.run(op, feed_dict: (input, np.array(1, 2, 3, 4, 5, 6).astype(NPTypeCode.Single) + 0.1f));

            ret.Should().BeOfType<long>().And.BeShaped(2, 3).And.BeOfValues(1, 2, 3, 4, 5, 6);
            print(ret.dtype);
            print(ret);
        }

        [TestMethod]
        public void Autocast_Case4()
        {
            var sess = tf.Session().as_default();
            var input = tf.placeholder(tf.byte8, shape: new TensorShape(6));
            var op = tf.reshape(input, new int[] { 2, 3 });
            sess.run(tf.global_variables_initializer());
            var ret = sess.run(op, feed_dict: (input, np.array(1, 2, 3, 4, 5, 6).astype(NPTypeCode.Single) + 0.1f));

            ret.Should().BeOfType<byte>().And.BeShaped(2, 3).And.BeOfValues(1, 2, 3, 4, 5, 6);
            print(ret.dtype);
            print(ret);
        }
    }
}