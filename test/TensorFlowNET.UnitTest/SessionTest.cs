using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using System;
using System.Collections.Generic;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Text;
using FluentAssertions;
using Google.Protobuf;
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
                    var result = (string) c.eval(sess);
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
                    var result = (string) c.eval(sess);
                    Console.WriteLine((string) result);
                    result.Should().HaveLength(size - 5000).And.ContainAll("a");
                }
            }
        }
    }
}