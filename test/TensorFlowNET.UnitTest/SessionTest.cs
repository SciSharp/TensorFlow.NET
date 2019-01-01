using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow;

namespace TensorFlowNET.UnitTest
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
            var s = new Status();
            var graph = new Graph();

            // Make a placeholder operation.
            var feed = c_test_util.ScalarConst(3, graph, s, "scalar1"); //c_test_util.Placeholder(graph, s);

            // Make a constant operation with the scalar "2".
            var two = c_test_util.ScalarConst(2, graph, s, "scalar2");

            // Add operation.
            var add = c_test_util.Add(feed, two, graph, s);

            var csession = new CSession(graph, s);
            ASSERT_EQ(TF_Code.TF_OK, s.Code);

            // Run the graph.
            var inputs = new Dictionary<IntPtr, IntPtr>();
            inputs.Add(feed, c_test_util.Int32Tensor(3));
            //csession.SetInputs(inputs);

            var outputs = new List<IntPtr> { add };
            csession.SetOutputs(outputs);

            csession.Run(s);
            Tensor outTensor = csession.output_tensor(0);
            EXPECT_EQ(TF_DataType.TF_INT32, outTensor.dtype);
            EXPECT_EQ(0, outTensor.NDims);
            ASSERT_EQ((ulong)sizeof(uint), outTensor.bytesize);
            var output_contents = outTensor.Data<int>();
            EXPECT_EQ(3 + 2, output_contents[0]);
        }
    }
}
