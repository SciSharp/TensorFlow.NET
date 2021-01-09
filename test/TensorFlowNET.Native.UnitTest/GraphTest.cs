using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using static Tensorflow.Binding;

namespace Tensorflow.Native.UnitTest
{
    [TestClass]
    public class GraphTest : CApiTest
    {
        /// <summary>
        /// Port from c_api_test.cc
        /// `TEST(CAPI, Graph)`
        /// </summary>
        [TestMethod]
        public void Graph()
        {
            var s = new Status();
            var graph = new Graph();

            // Make a placeholder operation.
            var feed = c_test_util.Placeholder(graph, s);
            EXPECT_EQ("feed", feed.name);
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
            EXPECT_EQ(-1, c_api.TF_OperationOutputListLength(feed, "bogus", s.Handle));
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
            EXPECT_EQ("add", add.name);
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
            EXPECT_EQ(2, (int)attr_value.I);

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
            var node_def = neg.node_def;

            // Validate NodeDef is what we expect.
            ASSERT_TRUE(c_test_util.IsNeg(node_def, "add"));

            // Serialize to GraphDef.
            var graph_def2 = c_test_util.GetGraphDef(graph);

            // Compare with first GraphDef + added NodeDef.
            graph_def.Node.Add(node_def);
            EXPECT_EQ(graph_def, graph_def2);

            // Look up some nodes by name.
            Operation neg2 = c_api.TF_GraphOperationByName(graph, "neg");
            EXPECT_EQ(neg, neg2);
            var node_def2 = neg2.node_def;
            EXPECT_EQ(node_def, node_def2);

            Operation feed2 = c_api.TF_GraphOperationByName(graph, "feed");
            EXPECT_EQ(feed, feed2);
            node_def = feed.node_def;
            node_def2 = feed2.node_def;
            EXPECT_EQ(node_def, node_def2);

            // Test iterating through the nodes of a graph.
            found_placeholder = false;
            found_scalar_const = false;
            found_add = false;
            bool found_neg = false;
            uint pos = 0;
            Operation oper;

            while ((oper = c_api.TF_GraphNextOperation(graph, ref pos)) != IntPtr.Zero)
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
                    node_def = oper.node_def;
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
        public void ImportGraphDef()
        {
            var s = new Status();
            var graph = new Graph().as_default();

            // Create a simple graph.
            c_test_util.Placeholder(graph, s);
            var oper = c_test_util.ScalarConst(3, graph, s);
            c_test_util.Neg(oper, graph, s);

            // Export to a GraphDef.
            var graph_def = new Buffer();
            c_api.TF_GraphToGraphDef(graph, graph_def.Handle, s.Handle);
            EXPECT_EQ(TF_Code.TF_OK, s.Code);

            // Import it, with a prefix, in a fresh graph.
            graph.Dispose();
            graph = new Graph().as_default();
            using (var opts = c_api.TF_NewImportGraphDefOptions())
            {
                c_api.TF_ImportGraphDefOptionsSetPrefix(opts, "imported");
                c_api.TF_GraphImportGraphDef(graph, graph_def.Handle, opts, s.Handle);
                EXPECT_EQ(TF_Code.TF_OK, s.Code);
            }

            Operation scalar = graph.OperationByName("imported/scalar");
            Operation feed = graph.OperationByName("imported/feed");
            Operation neg = graph.OperationByName("imported/neg");

            // Test basic structure of the imported graph.
            EXPECT_EQ(0, scalar.NumInputs);
            EXPECT_EQ(0, feed.NumInputs);
            EXPECT_EQ(1, neg.NumInputs);

            var neg_input = neg.Input(0);
            EXPECT_EQ(scalar, neg_input.oper);
            EXPECT_EQ(0, neg_input.index);

            // Test that we can't see control edges involving the source and sink nodes.
            EXPECT_EQ(0, scalar.NumControlInputs);
            EXPECT_EQ(0, scalar.GetControlInputs().Length);
            EXPECT_EQ(0, scalar.NumControlOutputs);
            EXPECT_EQ(0, scalar.GetControlOutputs().Length);

            EXPECT_EQ(0, feed.NumControlInputs);
            EXPECT_EQ(0, feed.GetControlInputs().Length);
            EXPECT_EQ(0, feed.NumControlOutputs);
            EXPECT_EQ(0, feed.GetControlOutputs().Length);

            EXPECT_EQ(0, neg.NumControlInputs);
            EXPECT_EQ(0, neg.GetControlInputs().Length);
            EXPECT_EQ(0, neg.NumControlOutputs);
            EXPECT_EQ(0, neg.GetControlOutputs().Length);

            static SafeImportGraphDefResultsHandle ImportGraph(Status s, Graph graph, Buffer graph_def, Operation scalar)
            {
                using var opts = c_api.TF_NewImportGraphDefOptions();
                c_api.TF_ImportGraphDefOptionsSetPrefix(opts, "imported2");
                c_api.TF_ImportGraphDefOptionsAddInputMapping(opts, "scalar", 0, new TF_Output(scalar, 0));
                c_api.TF_ImportGraphDefOptionsAddReturnOutput(opts, "feed", 0);
                c_api.TF_ImportGraphDefOptionsAddReturnOutput(opts, "scalar", 0);
                EXPECT_EQ(2, c_api.TF_ImportGraphDefOptionsNumReturnOutputs(opts));
                c_api.TF_ImportGraphDefOptionsAddReturnOperation(opts, "scalar");
                EXPECT_EQ(1, c_api.TF_ImportGraphDefOptionsNumReturnOperations(opts));
                var results = c_api.TF_GraphImportGraphDefWithResults(graph, graph_def.Handle, opts, s.Handle);
                EXPECT_EQ(TF_Code.TF_OK, s.Code);

                return results;
            }

            // Import it again, with an input mapping, return outputs, and a return
            // operation, into the same graph.
            Operation feed2;
            using (SafeImportGraphDefResultsHandle results = ImportGraph(s, graph, graph_def, scalar))
            {
                Operation scalar2 = graph.OperationByName("imported2/scalar");
                feed2 = graph.OperationByName("imported2/feed");
                Operation neg2 = graph.OperationByName("imported2/neg");

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
                var return_opers = graph.ReturnOperations(results);
                ASSERT_EQ(1, return_opers.Length);
                EXPECT_EQ(scalar2, return_opers[0]);  // not remapped
            }

            // Import again, with control dependencies, into the same graph.
            using (var opts = c_api.TF_NewImportGraphDefOptions())
            {
                c_api.TF_ImportGraphDefOptionsSetPrefix(opts, "imported3");
                c_api.TF_ImportGraphDefOptionsAddControlDependency(opts, feed);
                c_api.TF_ImportGraphDefOptionsAddControlDependency(opts, feed2);
                c_api.TF_GraphImportGraphDef(graph, graph_def.Handle, opts, s.Handle);
                EXPECT_EQ(TF_Code.TF_OK, s.Code);
            }

            var scalar3 = graph.OperationByName("imported3/scalar");
            var feed3 = graph.OperationByName("imported3/feed");
            var neg3 = graph.OperationByName("imported3/neg");
            ASSERT_TRUE(scalar3 != IntPtr.Zero);
            ASSERT_TRUE(feed3 != IntPtr.Zero);
            ASSERT_TRUE(neg3 != IntPtr.Zero);

            // Check that newly-imported scalar and feed have control deps (neg3 will
            // inherit them from input)
            var control_inputs = scalar3.GetControlInputs();
            ASSERT_EQ(2, scalar3.NumControlInputs);
            EXPECT_EQ(feed, control_inputs[0]);
            EXPECT_EQ(feed2, control_inputs[1]);

            control_inputs = feed3.GetControlInputs();
            ASSERT_EQ(2, feed3.NumControlInputs);
            EXPECT_EQ(feed, control_inputs[0]);
            EXPECT_EQ(feed2, control_inputs[1]);

            // Export to a graph def so we can import a graph with control dependencies
            graph_def = new Buffer();
            c_api.TF_GraphToGraphDef(graph, graph_def.Handle, s.Handle);
            EXPECT_EQ(TF_Code.TF_OK, s.Code);

            // Import again, with remapped control dependency, into the same graph
            using (var opts = c_api.TF_NewImportGraphDefOptions())
            {
                c_api.TF_ImportGraphDefOptionsSetPrefix(opts, "imported4");
                c_api.TF_ImportGraphDefOptionsRemapControlDependency(opts, "imported/feed", feed);
                c_api.TF_GraphImportGraphDef(graph, graph_def.Handle, opts, s.Handle);
                ASSERT_EQ(TF_Code.TF_OK, s.Code);
            }

            var scalar4 = graph.OperationByName("imported4/imported3/scalar");
            var feed4 = graph.OperationByName("imported4/imported2/feed");

            // Check that imported `imported3/scalar` has remapped control dep from
            // original graph and imported control dep
            control_inputs = scalar4.GetControlInputs();
            ASSERT_EQ(2, scalar4.NumControlInputs);
            EXPECT_EQ(feed, control_inputs[0]);
            EXPECT_EQ(feed4, control_inputs[1]);

            // Can add nodes to the imported graph without trouble.
            c_test_util.Add(feed, scalar, graph, s);
            ASSERT_EQ(TF_Code.TF_OK, s.Code);
        }

        /// <summary>
        /// Port from c_api_test.cc
        /// `TEST(CAPI, ImportGraphDef_WithReturnOutputs)`
        /// </summary>
        [TestMethod]
        public void ImportGraphDef_WithReturnOutputs()
        {
            var s = new Status();
            var graph = new Graph().as_default();

            // Create a graph with two nodes: x and 3
            c_test_util.Placeholder(graph, s);
            ASSERT_TRUE(graph.OperationByName("feed") != null);
            var oper = c_test_util.ScalarConst(3, graph, s);
            ASSERT_TRUE(graph.OperationByName("scalar") != null);
            c_test_util.Neg(oper, graph, s);
            ASSERT_TRUE(graph.OperationByName("neg") != null);

            // Export to a GraphDef.
            var graph_def = graph.ToGraphDef(s);
            ASSERT_EQ(TF_Code.TF_OK, s.Code);

            // Import it in a fresh graph with return outputs.
            graph.Dispose();
            graph = new Graph().as_default();
            var opts = new ImportGraphDefOptions();
            opts.AddReturnOutput("feed", 0);
            opts.AddReturnOutput("scalar", 0);
            EXPECT_EQ(2, opts.NumReturnOutputs);
            var return_outputs = graph.ImportGraphDefWithReturnOutputs(graph_def, opts, s);
            ASSERT_EQ(TF_Code.TF_OK, s.Code);

            var scalar = graph.OperationByName("scalar");
            var feed = graph.OperationByName("feed");
            var neg = graph.OperationByName("neg");
            ASSERT_TRUE(scalar != IntPtr.Zero);
            ASSERT_TRUE(feed != IntPtr.Zero);
            ASSERT_TRUE(neg != IntPtr.Zero);

            // Check return outputs
            EXPECT_EQ(feed, return_outputs[0].oper);
            EXPECT_EQ(0, return_outputs[0].index);
            EXPECT_EQ(scalar, return_outputs[1].oper);
            EXPECT_EQ(0, return_outputs[1].index);

            opts.Dispose();
            graph_def.Dispose();
            graph.Dispose();
            s.Dispose();
        }

        /// <summary>
        /// `TEST(CAPI, ImportGraphDef_MissingUnusedInputMappings)`
        /// </summary>
        [TestMethod]
        public void ImportGraphDef_MissingUnusedInputMappings()
        {

        }

        [Ignore]
        [TestMethod]
        public void ImportGraphMeta()
        {
            var dir = "my-save-dir/";
            using (var sess = tf.Session())
            {
                var new_saver = tf.train.import_meta_graph(dir + "my-model-10000.meta");
                new_saver.restore(sess, dir + "my-model-10000");
                var labels = tf.constant(0, dtype: tf.int32, shape: new int[] { 100 }, name: "labels");
                var batch_size = tf.size(labels);
                var logits = tf.get_collection<ITensorOrOperation>("logits")[0] as Tensor;
                var loss = tf.losses.sparse_softmax_cross_entropy(labels: labels,
                                                logits: logits);
            }
        }
    }
}
