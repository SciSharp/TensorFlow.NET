using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using System;
using System.Collections.Generic;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Text;
using FluentAssertions;
using Google.Protobuf;
using NumSharp.Backends;
using Tensorflow;
using Tensorflow.Util;
using static Tensorflow.Binding;

namespace TensorFlowNET.UnitTest
{
    [TestClass]
    public class SessionTest : CApiTest
    {
        /// <summary>
        /// tensorflow\c\c_api_test.cc
        /// `TEST(CAPI, Session)`
        /// </summary>
        [TestMethod, Ignore]
        public void Session()
        {
            lock (Locks.ProcessWide)
            {
                var s = new Status();
                var graph = new Graph().as_default();

                // Make a placeholder operation.
                var feed = c_test_util.Placeholder(graph, s);

                // Make a constant operation with the scalar "2".
                var two = c_test_util.ScalarConst(2, graph, s);

                // Add operation.
                var add = c_test_util.Add(feed, two, graph, s);

                var csession = new CSession(graph, s);
                ASSERT_EQ(TF_Code.TF_OK, s.Code);

                // Run the graph.
                var inputs = new Dictionary<Operation, Tensor>();
                inputs.Add(feed, new Tensor(3));
                csession.SetInputs(inputs);

                var outputs = new TF_Output[] {new TF_Output(add, 0)};
                csession.SetOutputs(outputs);

                csession.Run(s);
                Tensor outTensor = csession.output_tensor(0);
                EXPECT_EQ(TF_DataType.TF_INT32, outTensor.dtype);
                EXPECT_EQ(0, outTensor.NDims);
                ASSERT_EQ((ulong) sizeof(uint), outTensor.bytesize);
                var output_contents = outTensor.ToArray<int>();
                EXPECT_EQ(3 + 2, output_contents[0]);

                // Add another operation to the graph.
                var neg = c_test_util.Neg(add, graph, s);
                ASSERT_EQ(TF_Code.TF_OK, s.Code);

                // Run up to the new operation.
                inputs = new Dictionary<Operation, Tensor>();
                inputs.Add(feed, new Tensor(7));
                csession.SetInputs(inputs);
                outputs = new TF_Output[] {new TF_Output(neg, 0)};
                csession.SetOutputs(outputs);
                csession.Run(s);
                ASSERT_EQ(TF_Code.TF_OK, s.Code);

                outTensor = csession.output_tensor(0);
                ASSERT_TRUE(outTensor != IntPtr.Zero);
                EXPECT_EQ(TF_DataType.TF_INT32, outTensor.dtype);
                EXPECT_EQ(0, outTensor.NDims); // scalar
                ASSERT_EQ((ulong) sizeof(uint), outTensor.bytesize);
                output_contents = outTensor.ToArray<int>();
                EXPECT_EQ(-(7 + 2), output_contents[0]);

                // Clean up
                csession.CloseAndDelete(s);
                ASSERT_EQ(TF_Code.TF_OK, s.Code);
            }
        }

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
        public void Autocast_Case1()
        {
            var sess = tf.Session().as_default();
            var input = tf.placeholder(tf.float32, shape: new TensorShape(6));
            var op = tf.reshape(input, new int[] {2, 3});
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
            var op = tf.reshape(input, new int[] {2, 3});
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
            var op = tf.reshape(input, new int[] {2, 3});
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
            var op = tf.reshape(input, new int[] {2, 3});
            sess.run(tf.global_variables_initializer());
            var ret = sess.run(op, feed_dict: (input, np.array(1, 2, 3, 4, 5, 6).astype(NPTypeCode.Single) + 0.1f));

            ret.Should().BeOfType<byte>().And.BeShaped(2, 3).And.BeOfValues(1, 2, 3, 4, 5, 6);
            print(ret.dtype);
            print(ret);
        }
    }
}