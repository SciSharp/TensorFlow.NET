using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow;

namespace TensorFlowNET.UnitTest
{
    [TestClass]
    public class GraphTest
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
            Assert.AreEqual("feed", feed.Name);
            Assert.AreEqual("Placeholder", feed.OpType);
            Assert.AreEqual("", feed.Device);
            Assert.AreEqual(1, feed.NumOutputs);
            Assert.AreEqual(TF_DataType.TF_INT32, feed.OutputType(0));
            Assert.AreEqual(1, feed.OutputListLength("output"));
            Assert.AreEqual(0, feed.NumInputs);
            Assert.AreEqual(0, feed.OutputNumConsumers(0));
            Assert.AreEqual(0, feed.NumControlInputs);
            Assert.AreEqual(0, feed.NumControlOutputs);

            AttrValue attr_value = null;
            Assert.IsTrue(c_test_util.GetAttrValue(feed, "dtype", ref attr_value, s));
            Assert.AreEqual(attr_value.Type, DataType.DtInt32);

            // Test not found errors in TF_Operation*() query functions.
            Assert.AreEqual(-1, c_api.TF_OperationOutputListLength(feed, "bogus", s));
            Assert.AreEqual(TF_Code.TF_INVALID_ARGUMENT, s.Code);
            Assert.IsFalse(c_test_util.GetAttrValue(feed, "missing", ref attr_value, s));
            Assert.AreEqual("Operation 'feed' has no attr named 'missing'.", s.Message);

            // Make a constant oper with the scalar "3".
            var three = c_test_util.ScalarConst(3, graph, s);
            Assert.AreEqual(TF_Code.TF_OK, s.Code);

            // Add oper.
            var add = c_test_util.Add(feed, three, graph, s);
            Assert.AreEqual(TF_Code.TF_OK, s.Code);

            // Test TF_Operation*() query functions.
            Assert.AreEqual("add", add.Name);
            Assert.AreEqual("AddN", add.OpType);
            Assert.AreEqual("", add.Device);
            Assert.AreEqual(1, add.NumOutputs);
            Assert.AreEqual(TF_DataType.TF_INT32, add.OutputType(0));
            Assert.AreEqual(1, add.OutputListLength("sum"));
            Assert.AreEqual(TF_Code.TF_OK, s.Code);
            Assert.AreEqual(2, add.InputListLength("inputs"));
            Assert.AreEqual(TF_Code.TF_OK, s.Code);
            Assert.AreEqual(TF_DataType.TF_INT32, add.InputType(0));
            Assert.AreEqual(TF_DataType.TF_INT32, add.InputType(1));
            var add_in_0 = add.Input(0);
            Assert.AreEqual(feed, add_in_0.oper);
            Assert.AreEqual(0, add_in_0.index);
            var add_in_1 = add.Input(1);
            Assert.AreEqual(three, add_in_1.oper);
            Assert.AreEqual(0, add_in_1.index);
            Assert.AreEqual(0, add.OutputNumConsumers(0));
            Assert.AreEqual(0, add.NumControlInputs);
            Assert.AreEqual(0, add.NumControlOutputs);

            Assert.IsTrue(c_test_util.GetAttrValue(add, "T", ref attr_value, s));
            Assert.AreEqual(DataType.DtInt32, attr_value.Type);
            Assert.IsTrue(c_test_util.GetAttrValue(add, "N", ref attr_value, s));
            Assert.AreEqual(2, attr_value.I);

            // Placeholder oper now has a consumer.
            Assert.AreEqual(1, feed.OutputNumConsumers(0));
            TF_Input[] feed_port = feed.OutputConsumers(0, 1);
            Assert.AreEqual(1, feed_port.Length);
            Assert.AreEqual(add, feed_port[0].oper);
            Assert.AreEqual(0, feed_port[0].index);

            // The scalar const oper also has a consumer.
            Assert.AreEqual(1, three.OutputNumConsumers(0));
            TF_Input[] three_port = three.OutputConsumers(0, 1);
            Assert.AreEqual(add, three_port[0].oper);
            Assert.AreEqual(1, three_port[0].index);

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
            Assert.IsTrue(found_placeholder);
            Assert.IsTrue(found_scalar_const);
            Assert.IsTrue(found_add);

            // Add another oper to the graph.
            var neg = c_test_util.Neg(add, graph, s);
            Assert.AreEqual(TF_Code.TF_OK, s.Code);

            // Serialize to NodeDef.
            var node_def = c_test_util.GetNodeDef(neg);

            // Validate NodeDef is what we expect.
            Assert.IsTrue(c_test_util.IsNeg(node_def, "add"));

            // Serialize to GraphDef.
            var graph_def2 = c_test_util.GetGraphDef(graph);

            // Compare with first GraphDef + added NodeDef.
            graph_def.Node.Add(node_def);
            Assert.AreEqual(graph_def.ToString(), graph_def2.ToString());

            // Look up some nodes by name.
            Operation neg2 = c_api.TF_GraphOperationByName(graph, "neg");
            Assert.AreEqual(neg, neg2);
            var node_def2 = c_test_util.GetNodeDef(neg2);
            Assert.AreEqual(node_def.ToString(), node_def2.ToString());

            Operation feed2 = c_api.TF_GraphOperationByName(graph, "feed");
            Assert.AreEqual(feed, feed2);
            node_def = c_test_util.GetNodeDef(feed);
            node_def2 = c_test_util.GetNodeDef(feed2);
            Assert.AreEqual(node_def.ToString(), node_def2.ToString());

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

            Assert.IsTrue(found_placeholder);
            Assert.IsTrue(found_scalar_const);
            Assert.IsTrue(found_add);
            Assert.IsTrue(found_neg);

            graph.Dispose();
            s.Dispose();
        }
    }
}
