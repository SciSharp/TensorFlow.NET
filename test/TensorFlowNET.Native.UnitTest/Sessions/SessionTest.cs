using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;

namespace Tensorflow.Native.UnitTest.Sessions
{
    [TestClass]
    public class SessionTest : CApiTest
    {
        /// <summary>
        /// tensorflow\c\c_api_test.cc
        /// `TEST(CAPI, Session)`
        /// </summary>
        [TestMethod]
        public void Session()
        {
            using var s = new Status();
            using var graph = new Graph();

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

            var outputs = new TF_Output[] { new TF_Output(add, 0) };
            csession.SetOutputs(outputs);

            csession.Run(s);
            Tensor outTensor = csession.output_tensor(0);
            EXPECT_EQ(TF_DataType.TF_INT32, outTensor.dtype);
            EXPECT_EQ(0, outTensor.NDims);
            ASSERT_EQ((ulong)sizeof(uint), outTensor.bytesize);
            var output_contents = outTensor.ToArray<int>();
            EXPECT_EQ(3 + 2, output_contents[0]);

            // Add another operation to the graph.
            var neg = c_test_util.Neg(add, graph, s);
            ASSERT_EQ(TF_Code.TF_OK, s.Code);

            // Run up to the new operation.
            inputs = new Dictionary<Operation, Tensor>();
            inputs.Add(feed, new Tensor(7));
            csession.SetInputs(inputs);
            outputs = new TF_Output[] { new TF_Output(neg, 0) };
            csession.SetOutputs(outputs);
            csession.Run(s);
            ASSERT_EQ(TF_Code.TF_OK, s.Code);

            outTensor = csession.output_tensor(0);
            ASSERT_TRUE(outTensor != IntPtr.Zero);
            EXPECT_EQ(TF_DataType.TF_INT32, outTensor.dtype);
            EXPECT_EQ(0, outTensor.NDims); // scalar
            ASSERT_EQ((ulong)sizeof(uint), outTensor.bytesize);
            output_contents = outTensor.ToArray<int>();
            EXPECT_EQ(-(7 + 2), output_contents[0]);

            // Clean up
            csession.CloseAndDelete(s);
            ASSERT_EQ(TF_Code.TF_OK, s.Code);
        }
    }
}
