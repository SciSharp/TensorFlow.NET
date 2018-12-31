using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;
using Tensorflow;
using Buffer = Tensorflow.Buffer;

namespace TensorFlowNET.UnitTest
{
    [TestClass]
    public class GraphTest : CApiTest
    {
        /// <summary>
        /// Port from c_api_test.cc
        /// `TEST(CAPI, Graph)`
        /// </summary>
        [TestMethod]
        public void c_api_Graph()
        {
            var s = new Status();
            var graph = new Graph();

            // Make a placeholder operation.
            var feed = c_test_util.Placeholder(graph, s);
            EXPECT_EQ("feed", feed.Name);
            EXPECT_EQ("Placeholder", feed.OpType);
            EXPECT_EQ("", feed.Device);
            EXPECT_EQ(1, feed.NumOutputs);
            EXPECT_EQ(TF_DataType.TF_INT32, feed.OutputType(0));
            EXPECT_EQ(1, feed.OutputListLength("output"));
            EXPECT_EQ(0, feed.NumInputs);
            EXPECT_EQ(0, feed.OutputNumConsumers(0));
            EXPECT_EQ(0, feed.NumControlInputs);
            EXPECT_EQ(0, feed.NumControlOutputs);

            AttrValue attr_value = null;
            ASSERT_TRUE(c_test_util.GetAttrValue(feed, "dtype", ref attr_value, s));
            EXPECT_EQ(attr_value.Type, DataType.DtInt32);

            // Test not found errors in TF_Operation*() query functions.
            EXPECT_EQ(-1, c_api.TF_OperationOutputListLength(feed, "bogus", s));
            EXPECT_EQ(TF_Code.TF_INVALID_ARGUMENT, s.Code);
            Assert.IsFalse(c_test_util.GetAttrValue(feed, "missing", ref attr_value, s));
            EXPECT_EQ("Operation 'feed' has no attr named 'missing'.", s.Message);

            // Make a constant oper with the scalar "3".
            var three = c_test_util.ScalarConst(3, graph, s);
            EXPECT_EQ(TF_Code.TF_OK, s.Code);

            // Add oper.
            var add = c_test_util.Add(feed, three, graph, s);
            EXPECT_EQ(TF_Code.TF_OK, s.Code);

            // Test TF_Operation*() query functions.
            EXPECT_EQ("add", add.Name);
            EXPECT_EQ("AddN", add.OpType);
            EXPECT_EQ("", add.Device);
            EXPECT_EQ(1, add.NumOutputs);
            EXPECT_EQ(TF_DataType.TF_INT32, add.OutputType(0));
            EXPECT_EQ(1, add.OutputListLength("sum"));
            EXPECT_EQ(TF_Code.TF_OK, s.Code);
            EXPECT_EQ(2, add.InputListLength("inputs"));
            EXPECT_EQ(TF_Code.TF_OK, s.Code);
            EXPECT_EQ(TF_DataType.TF_INT32, add.InputType(0));
            EXPECT_EQ(TF_DataType.TF_INT32, add.InputType(1));
            var add_in_0 = add.Input(0);
            EXPECT_EQ(feed, add_in_0.oper);
            EXPECT_EQ(0, add_in_0.index);
            var add_in_1 = add.Input(1);
            EXPECT_EQ(three, add_in_1.oper);
            EXPECT_EQ(0, add_in_1.index);
            EXPECT_EQ(0, add.OutputNumConsumers(0));
            EXPECT_EQ(0, add.NumControlInputs);
            EXPECT_EQ(0, add.NumControlOutputs);

            ASSERT_TRUE(c_test_util.GetAttrValue(add, "T", ref attr_value, s));
            EXPECT_EQ(DataType.DtInt32, attr_value.Type);
            ASSERT_TRUE(c_test_util.GetAttrValue(add, "N", ref attr_value, s));
            EXPECT_EQ(2, attr_value.I);

            // Placeholder oper now has a consumer.
            EXPECT_EQ(1, feed.OutputNumConsumers(0));
            TF_Input[] feed_port = feed.OutputConsumers(0, 1);
            EXPECT_EQ(1, feed_port.Length);
            EXPECT_EQ(add, feed_port[0].oper);
            EXPECT_EQ(0, feed_port[0].index);

            // The scalar const oper also has a consumer.
            EXPECT_EQ(1, three.OutputNumConsumers(0));
            TF_Input[] three_port = three.OutputConsumers(0, 1);
            EXPECT_EQ(add, three_port[0].oper);
            EXPECT_EQ(1, three_port[0].index);

            // Serialize to GraphDef.
            var graph_def = c_test_util.GetGraphDef(graph);

            // Validate GraphDef is what we expect.
            bool found_placeholder = false;
            bool found_scalar_const = false;
            bool found_add = false;
            foreach (var n in graph_def.Node)
            {
                if (c_test_util.IsPlaceholder(n))
                {
                    Assert.IsFalse(found_placeholder);
                    found_placeholder = true;
                }
                else if (c_test_util.IsScalarConst(n, 3))
                {
                    Assert.IsFalse(found_scalar_const);
                    found_scalar_const = true;
                }
                else if (c_test_util.IsAddN(n, 2))
                {
                    Assert.IsFalse(found_add);
                    found_add = true;
                }
                else
                {
                    Assert.Fail($"Unexpected NodeDef: {n}");
                }
            }
            ASSERT_TRUE(found_placeholder);
            ASSERT_TRUE(found_scalar_const);
            ASSERT_TRUE(found_add);

            // Add another oper to the graph.
            var neg = c_test_util.Neg(add, graph, s);
            EXPECT_EQ(TF_Code.TF_OK, s.Code);

            // Serialize to NodeDef.
            var node_def = c_test_util.GetNodeDef(neg);

            // Validate NodeDef is what we expect.
            ASSERT_TRUE(c_test_util.IsNeg(node_def, "add"));

            // Serialize to GraphDef.
            var graph_def2 = c_test_util.GetGraphDef(graph);

            // Compare with first GraphDef + added NodeDef.
            graph_def.Node.Add(node_def);
            EXPECT_EQ(graph_def.ToString(), graph_def2.ToString());

            // Look up some nodes by name.
            Operation neg2 = c_api.TF_GraphOperationByName(graph, "neg");
            EXPECT_EQ(neg, neg2);
            var node_def2 = c_test_util.GetNodeDef(neg2);
            EXPECT_EQ(node_def.ToString(), node_def2.ToString());

            Operation feed2 = c_api.TF_GraphOperationByName(graph, "feed");
            EXPECT_EQ(feed, feed2);
            node_def = c_test_util.GetNodeDef(feed);
            node_def2 = c_test_util.GetNodeDef(feed2);
            EXPECT_EQ(node_def.ToString(), node_def2.ToString());

            // Test iterating through the nodes of a graph.
            found_placeholder = false;
            found_scalar_const = false;
            found_add = false;
            bool found_neg = false;
            uint pos = 0;
            Operation oper;

            while((oper = c_api.TF_GraphNextOperation(graph, ref pos)) != IntPtr.Zero)
            {
                if (oper.Equals(feed))
                {
                    Assert.IsFalse(found_placeholder);
                    found_placeholder = true;
                }
                else if (oper.Equals(three))
                {
                    Assert.IsFalse(found_scalar_const);
                    found_scalar_const = true;
                }
                else if (oper.Equals(add))
                {
                    Assert.IsFalse(found_add);
                    found_add = true;
                }
                else if (oper.Equals(neg))
                {
                    Assert.IsFalse(found_neg);
                    found_neg = true;
                }
                else
                {
                    node_def = c_test_util.GetNodeDef(oper);
                    Assert.Fail($"Unexpected Node: {node_def.ToString()}");
                }
            }

            ASSERT_TRUE(found_placeholder);
            ASSERT_TRUE(found_scalar_const);
            ASSERT_TRUE(found_add);
            ASSERT_TRUE(found_neg);

            graph.Dispose();
            s.Dispose();
        }

        /// <summary>
        /// Port from c_api_test.cc
        /// `TEST(CAPI, ImportGraphDef)`
        /// </summary>
        [TestMethod]
        public void c_api_ImportGraphDef()
        {
            var s = new Status();
            var graph = new Graph();

            // Create a simple graph.
            c_test_util.Placeholder(graph, s);
            var oper = c_test_util.ScalarConst(3, graph, s);
            c_test_util.Neg(oper, graph, s);

            // Export to a GraphDef.
            var graph_def = new Buffer();
            c_api.TF_GraphToGraphDef(graph, graph_def, s);
            EXPECT_EQ(TF_Code.TF_OK, s.Code);

            // Import it, with a prefix, in a fresh graph.
            graph.Dispose();
            graph = new Graph();
            var opts = c_api.TF_NewImportGraphDefOptions();
            c_api.TF_ImportGraphDefOptionsSetPrefix(opts, "imported");
            c_api.TF_GraphImportGraphDef(graph, graph_def, opts, s);
            EXPECT_EQ(TF_Code.TF_OK, s.Code);

            Operation scalar = c_api.TF_GraphOperationByName(graph, "imported/scalar");
            Operation feed = c_api.TF_GraphOperationByName(graph, "imported/feed");
            Operation neg = c_api.TF_GraphOperationByName(graph, "imported/neg");

            // Test basic structure of the imported graph.
            EXPECT_EQ(0, scalar.NumInputs);
            EXPECT_EQ(0, feed.NumInputs);
            EXPECT_EQ(1, neg.NumInputs);

            var neg_input = neg.Input(0);
            EXPECT_EQ(scalar, neg_input.oper);
            EXPECT_EQ(0, neg_input.index);

            // Test that we can't see control edges involving the source and sink nodes.
            EXPECT_EQ(0, scalar.NumControlInputs);
            EXPECT_EQ(0, scalar.ControlInputs(100).Length);
            EXPECT_EQ(0, scalar.NumControlOutputs);
            EXPECT_EQ(0, scalar.ControlOutputs(100).Length);

            EXPECT_EQ(0, feed.NumControlInputs);
            EXPECT_EQ(0, feed.ControlInputs(100).Length);
            EXPECT_EQ(0, feed.NumControlOutputs);
            EXPECT_EQ(0, feed.ControlOutputs(100).Length);

            EXPECT_EQ(0, neg.NumControlInputs);
            EXPECT_EQ(0, neg.ControlInputs(100).Length);
            EXPECT_EQ(0, neg.NumControlOutputs);
            EXPECT_EQ(0, neg.ControlOutputs(100).Length);

            // Import it again, with an input mapping, return outputs, and a return
            // operation, into the same graph.
            c_api.TF_DeleteImportGraphDefOptions(opts);
            opts = c_api.TF_NewImportGraphDefOptions();
            c_api.TF_ImportGraphDefOptionsSetPrefix(opts, "imported2");
            c_api.TF_ImportGraphDefOptionsAddInputMapping(opts, "scalar", 0, new TF_Output(scalar, 0));
            c_api.TF_ImportGraphDefOptionsAddReturnOutput(opts, "feed", 0);
            c_api.TF_ImportGraphDefOptionsAddReturnOutput(opts, "scalar", 0);
            EXPECT_EQ(2, c_api.TF_ImportGraphDefOptionsNumReturnOutputs(opts));
            c_api.TF_ImportGraphDefOptionsAddReturnOperation(opts, "scalar");
            EXPECT_EQ(1, c_api.TF_ImportGraphDefOptionsNumReturnOperations(opts));
            var results = c_api.TF_GraphImportGraphDefWithResults(graph, graph_def, opts, s);
            EXPECT_EQ(TF_Code.TF_OK, s.Code);

            Operation scalar2 = c_api.TF_GraphOperationByName(graph, "imported2/scalar");
            Operation feed2 = c_api.TF_GraphOperationByName(graph, "imported2/feed");
            Operation neg2 = c_api.TF_GraphOperationByName(graph, "imported2/neg");

            // Check input mapping
            neg_input = neg.Input(0);
            EXPECT_EQ(scalar, neg_input.oper);
            EXPECT_EQ(0, neg_input.index);

            // Check return outputs
            var return_outputs = graph.ReturnOutputs(results);
            ASSERT_EQ(2, return_outputs.Length);
            EXPECT_EQ(feed2, return_outputs[0].oper);
            EXPECT_EQ(0, return_outputs[0].index);
            EXPECT_EQ(scalar, return_outputs[1].oper);  // remapped
            EXPECT_EQ(0, return_outputs[1].index);

            // Check return operation
            var num_return_opers = graph.ReturnOperations(results);
            ASSERT_EQ(1, num_return_opers);
        }
    }
}
