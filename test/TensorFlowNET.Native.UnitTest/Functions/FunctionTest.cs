using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Linq;
using static Tensorflow.Native.UnitTest.c_test_util;

namespace Tensorflow.Native.UnitTest
{
    /// <summary>
    /// tensorflow\c\c_api_function_test.cc
    /// `class CApiColocationTest`
    /// </summary>
    [TestClass]
    public class FunctionTest : CApiTest, IDisposable
    {
        Graph func_graph_;
        Graph host_graph_;
        string func_name_ = "MyFunc";
        string func_node_name_ = "MyFunc_0";
        Status s_;
        IntPtr func_;

        [TestInitialize]
        public void Initialize()
        {
            func_graph_ = new Graph();
            host_graph_ = new Graph();
            s_ = new Status();
        }

        [TestMethod]
        public void OneOp_ZeroInputs_OneOutput()
        {
            var c = ScalarConst(10, func_graph_, s_, "scalar10");
            // Define
            Define(-1, new Operation[0], new Operation[0], new[] { c }, new string[0]);

            // Use, run, and verify
            var func_op = Use(new Operation[0]);
            Run(new KeyValuePair<Operation, Tensor>[0], func_op, 10);
            VerifyFDef(new[] { "scalar10_0" },
                new List<IOSpec>(),
                new List<IOSpec> { new IOSpec("scalar10", DataType.DtInt32) },
                new List<EdgeSpec> { new EdgeSpec("scalar10_0:output:0", "scalar10") },
                new List<EdgeSpec>());
        }

        [TestMethod]
        public void OneOp_OneInput_OneOutput()
        {
            // Define
            var feed = Placeholder(func_graph_, s_);
            var neg = Neg(feed, func_graph_, s_);
            Define(-1, new Operation[0], new[] { feed }, new[] { neg }, new string[0]);

            // Use, run, and verify
            var func_feed = Placeholder(host_graph_, s_);
            var func_op = Use(new[] { func_feed });
            Run(new[] { new KeyValuePair<Operation, Tensor>(func_feed, Int32Tensor(3)) }, func_op, -3);
            VerifyFDef(new string[] { "neg_0" },
                new List<IOSpec> { new IOSpec("feed", DataType.DtInt32) },
                new List<IOSpec> { new IOSpec("neg", DataType.DtInt32) },
                new List<EdgeSpec> { new EdgeSpec("feed", "neg_0:0"), new EdgeSpec("neg_0:y:0", "neg") },
                new List<EdgeSpec>());
        }

        [TestMethod]
        public void OneOutput_OutputNames()
        {
            // Define
            var feed = Placeholder(func_graph_, s_);
            var neg = Neg(feed, func_graph_, s_);
            Define(-1,
                new Operation[0],
                new[] { feed },
                new[] { neg },
                new[] { "negated_num" });

            // Use, run, and verify
            var func_feed = Placeholder(host_graph_, s_);
            var func_op = Use(new[] { func_feed });
            Run(new[] { new KeyValuePair<Operation, Tensor>(func_feed, Int32Tensor(3)) }, func_op, -3);
            VerifyFDef(new string[] { "neg" },
                new List<IOSpec> { new IOSpec("feed", DataType.DtInt32) },
                new List<IOSpec> { new IOSpec("negated_num", DataType.DtInt32) },
                new List<EdgeSpec> { new EdgeSpec("feed", "neg:0"), new EdgeSpec("neg:y:0", "negated_num") },
                new List<EdgeSpec>());
        }

        [TestMethod]
        public void OutputNames_SameNameAsInput()
        {
            // Define
            var feed = Placeholder(func_graph_, s_, "negation");
            var neg = Neg(feed, func_graph_, s_, "neg");
            Define(-1,
                new Operation[0],
                new[] { feed },
                new[] { neg },
                new[] { "negation" });

            // Use, run, and verify
            var func_feed = Placeholder(host_graph_, s_);
            var func_op = Use(new[] { func_feed });
            Run(new[] { new KeyValuePair<Operation, Tensor>(func_feed, Int32Tensor(3)) }, func_op, -3);
            VerifyFDef(new string[] { "neg" },
                new List<IOSpec> { new IOSpec("negation_0", DataType.DtInt32) },
                new List<IOSpec> { new IOSpec("negation", DataType.DtInt32) },
                new List<EdgeSpec> { new EdgeSpec("negation_0", "neg:0"), new EdgeSpec("neg:y:0", "negation") },
                new List<EdgeSpec>());
        }

        [TestMethod]
        public void ZeroOps_Identity()
        {
            // Define
            var feed = Placeholder(func_graph_, s_);
            Define(-1,
                new Operation[0],
                new[] { feed },
                new[] { feed },
                new string[0]);

            // Use, run, and verify
            var func_feed = Placeholder(host_graph_, s_);
            var func_op = Use(new[] { func_feed });
            Run(new[] { new KeyValuePair<Operation, Tensor>(func_feed, Int32Tensor(3)) }, func_op, 3);
            VerifyFDef(new string[0],
                new List<IOSpec> { new IOSpec("feed_0", DataType.DtInt32) },
                new List<IOSpec> { new IOSpec("feed", DataType.DtInt32) },
                new List<EdgeSpec> { new EdgeSpec("feed_0", "feed") },
                new List<EdgeSpec>());
        }

        [TestMethod]
        public void ZeroOps_Permutation()
        {
            // Define
            var feed1 = Placeholder(func_graph_, s_, "feed1");
            var feed2 = Placeholder(func_graph_, s_, "feed2");
            Define(-1,
                null,
                new[] { feed1, feed2 },
                new[] { feed2, feed1 },
                null);

            // Use, run, and verify
            var two = ScalarConst(2, host_graph_, s_);
            var func_feed = Placeholder(host_graph_, s_);
            var func_op = Use(new[] { two, func_feed });
            Run(new[] { new KeyValuePair<Operation, Tensor>(func_feed, Int32Tensor(3)) },
                new[] { new TF_Output(func_op, 0), new TF_Output(func_op, 1) },
                new[] { 3, 2 });
            VerifyFDef(new string[0],
                new List<IOSpec> { new IOSpec("feed1_0"), new IOSpec("feed2_0") },
                new List<IOSpec> { new IOSpec("feed2"), new IOSpec("feed1") },
                new List<EdgeSpec> { new EdgeSpec("feed1_0", "feed1"), new EdgeSpec("feed2_0", "feed2") },
                new List<EdgeSpec>());
        }

        [TestMethod]
        public void ZeroOps_Permutation_OutputNames()
        {
            // Define
            var feed1 = Placeholder(func_graph_, s_, "feed1");
            var feed2 = Placeholder(func_graph_, s_, "feed2");
            Define(-1,
                null,
                new[] { feed1, feed2 },
                new[] { feed2, feed1 },
                new[] { "first", "second" });

            // Use, run, and verify
            var two = ScalarConst(2, host_graph_, s_);
            var func_feed = Placeholder(host_graph_, s_);
            var func_op = Use(new[] { two, func_feed });
            Run(new[] { new KeyValuePair<Operation, Tensor>(func_feed, Int32Tensor(3)) },
                new[] { new TF_Output(func_op, 0), new TF_Output(func_op, 1) },
                new[] { 3, 2 });
            VerifyFDef(new string[0],
                new List<IOSpec> { new IOSpec("feed1"), new IOSpec("feed2") },
                new List<IOSpec> { new IOSpec("first"), new IOSpec("second") },
                new List<EdgeSpec> { new EdgeSpec("feed1", "second"), new EdgeSpec("feed2", "first") },
                new List<EdgeSpec>());
        }

        [TestMethod]
        public void OneOp_TwoInputs_OneOutput()
        {
            // Define
            var feed1 = Placeholder(func_graph_, s_, "feed1");
            var feed2 = Placeholder(func_graph_, s_, "feed2");
            var add = Add(feed1, feed2, func_graph_, s_);
            Define(-1,
                null,
                new[] { feed1, feed2 },
                new[] { add },
                null);

            // Use, run, and verify
            var two = ScalarConst(2, host_graph_, s_);
            var func_feed = Placeholder(host_graph_, s_);
            var func_op = Use(new[] { two, func_feed });
            Run(new[] { new KeyValuePair<Operation, Tensor>(func_feed, Int32Tensor(3)) },
                func_op,
                2 + 3);
            VerifyFDef(new string[] { "add_0" },
                new List<IOSpec> { new IOSpec("feed1"), new IOSpec("feed2") },
                new List<IOSpec> { new IOSpec("add") },
                new List<EdgeSpec>
                {
                    new EdgeSpec("feed1", "add_0:0"),
                    new EdgeSpec("feed2", "add_0:1"),
                    new EdgeSpec("add_0:sum:0", "add")
                },
                new List<EdgeSpec>());
        }

        [TestMethod]
        public void OneOp_TwoInputs_ZeroOutputs()
        {
            // Define
            var feed1 = Placeholder(func_graph_, s_, "feed1");
            var feed2 = Placeholder(func_graph_, s_, "feed2");
            var add = Add(feed1, feed2, func_graph_, s_);
            Define(-1,
                null,
                new[] { feed1, feed2 },
                new Operation[0],
                null);

            // Use, run, and verify
            var two = ScalarConst(2, host_graph_, s_);
            var func_feed = Placeholder(host_graph_, s_);
            var func_op = Use(new[] { two, func_feed });
            VerifyFDef(new string[] { "add" },
                new List<IOSpec> { new IOSpec("feed1"), new IOSpec("feed2") },
                new List<IOSpec>(),
                new List<EdgeSpec>
                {
                    new EdgeSpec("feed1", "add:0"),
                    new EdgeSpec("feed2", "add:1")
                },
                new List<EdgeSpec>());
        }

        [TestMethod]
        public void TwoOps_ThreeInputs_OneOutput()
        {
            // Define
            var feed1 = Placeholder(func_graph_, s_, "feed1");
            var feed2 = Placeholder(func_graph_, s_, "feed2");
            var feed3 = Placeholder(func_graph_, s_, "feed3");
            var add1 = Add(feed1, feed2, func_graph_, s_, "add1");
            var add2 = Add(add1, feed3, func_graph_, s_, "add2");
            Define(-1,
                null,
                new[] { feed1, feed2, feed3 },
                new[] { add2 },
                null);

            // Use, run, and verify
            var two = ScalarConst(2, host_graph_, s_, "two");
            var ten = ScalarConst(10, host_graph_, s_, "ten");
            var func_feed = Placeholder(host_graph_, s_);
            var func_op = Use(new[] { two, ten, func_feed });
            Run(new[] { new KeyValuePair<Operation, Tensor>(func_feed, Int32Tensor(3)) },
                func_op,
                2 + 10 + 3);
            VerifyFDef(new string[] { "add1", "add2_0" },
                new List<IOSpec> { new IOSpec("feed1"), new IOSpec("feed2"), new IOSpec("feed3") },
                new List<IOSpec> { new IOSpec("add2") },
                new List<EdgeSpec>
                {
                    new EdgeSpec("feed1", "add1:0"),
                    new EdgeSpec("feed2", "add1:1"),
                    new EdgeSpec("add1:sum:0", "add2_0:0"),
                    new EdgeSpec("feed3", "add2_0:1"),
                    new EdgeSpec("add2_0:sum:0", "add2"),
                },
                new List<EdgeSpec>());
        }

        [TestMethod]
        public void OneOp_TwoInputs_TwoDuplicateOutputs()
        {
            // Define
            var feed1 = Placeholder(func_graph_, s_, "feed1");
            var feed2 = Placeholder(func_graph_, s_, "feed2");
            var add = Add(feed1, feed2, func_graph_, s_);
            Define(-1,
                null,
                new[] { feed1, feed2 },
                new[] { add, add },
                null);

            // Use, run, and verify
            var two = ScalarConst(2, host_graph_, s_);
            var func_feed = Placeholder(host_graph_, s_);
            var func_op = Use(new[] { two, func_feed });
            Run(new[] { new KeyValuePair<Operation, Tensor>(func_feed, Int32Tensor(3)) },
                new[] { new TF_Output(func_op, 0), new TF_Output(func_op, 1) },
                new[] { 5, 5 });
            VerifyFDef(new string[] { "add_1" },
                new List<IOSpec> { new IOSpec("feed1"), new IOSpec("feed2") },
                new List<IOSpec> { new IOSpec("add"), new IOSpec("add_0") },
                new List<EdgeSpec>
                {
                    new EdgeSpec("feed1", "add_1:0"),
                    new EdgeSpec("feed2", "add_1:1"),
                    new EdgeSpec("add_1:sum:0", "add"),
                    new EdgeSpec("add_1:sum:0", "add_0")
                },
                new List<EdgeSpec>());
        }

        [TestMethod]
        public void TwoDuplicateOutputs_OutputNames()
        {
            // Define
            var feed1 = Placeholder(func_graph_, s_, "feed1");
            var feed2 = Placeholder(func_graph_, s_, "feed2");
            var add = Add(feed1, feed2, func_graph_, s_);
            Define(-1,
                null,
                new[] { feed1, feed2 },
                new[] { add, add },
                new[] { "out1", "out2" });

            // Use, run, and verify
            var two = ScalarConst(2, host_graph_, s_);
            var func_feed = Placeholder(host_graph_, s_);
            var func_op = Use(new[] { two, func_feed });
            Run(new[] { new KeyValuePair<Operation, Tensor>(func_feed, Int32Tensor(3)) },
                new[] { new TF_Output(func_op, 0), new TF_Output(func_op, 1) },
                new[] { 5, 5 });
            VerifyFDef(new string[] { "add" },
                new List<IOSpec> { new IOSpec("feed1"), new IOSpec("feed2") },
                new List<IOSpec> { new IOSpec("out1"), new IOSpec("out2") },
                new List<EdgeSpec>
                {
                    new EdgeSpec("feed1", "add:0"),
                    new EdgeSpec("feed2", "add:1"),
                    new EdgeSpec("add:sum:0", "out1"),
                    new EdgeSpec("add:sum:0", "out2")
                },
                new List<EdgeSpec>());
        }

        [TestMethod]
        public void TwoOps_ThreeInputs_TwoOutputs()
        {
            // Define
            var feed1 = Placeholder(func_graph_, s_, "feed1");
            var feed2 = Placeholder(func_graph_, s_, "feed2");
            var feed3 = Placeholder(func_graph_, s_, "feed3");
            var add1 = Add(feed1, feed2, func_graph_, s_, "add1");
            var add2 = Add(add1, feed3, func_graph_, s_, "add2");
            Define(-1,
                null,
                new[] { feed1, feed2, feed3 },
                new[] { add1, add2 },
                null);

            // Use, run, and verify
            var two = ScalarConst(2, host_graph_, s_, "two");
            var ten = ScalarConst(10, host_graph_, s_, "ten");
            var func_feed = Placeholder(host_graph_, s_);
            var func_op = Use(new[] { two, ten, func_feed });
            Run(new[] { new KeyValuePair<Operation, Tensor>(func_feed, Int32Tensor(3)) },
                new[] { new TF_Output(func_op, 0), new TF_Output(func_op, 1) },
                new[] { 12, 15 });
            VerifyFDef(new string[] { "add1_0", "add2_0" },
                new List<IOSpec> { new IOSpec("feed1"), new IOSpec("feed2"), new IOSpec("feed3") },
                new List<IOSpec> { new IOSpec("add1"), new IOSpec("add2") },
                new List<EdgeSpec>
                {
                    new EdgeSpec("feed1", "add1_0:0"),
                    new EdgeSpec("feed2", "add1_0:1"),
                    new EdgeSpec("add1_0:sum:0", "add2_0:0"),
                    new EdgeSpec("feed3", "add2_0:1"),
                    new EdgeSpec("add1_0:sum:0", "add1"),
                    new EdgeSpec("add2_0:sum:0", "add2")
                },
                new List<EdgeSpec>());
        }

        void Define(int num_opers, Operation[] opers,
            Operation[] inputs, Operation[] outputs,
            string[] output_names, bool expect_failure = false)
            => DefineT(num_opers, opers,
                inputs.Select(x => new TF_Output(x, 0)).ToArray(),
                outputs.Select(x => new TF_Output(x, 0)).ToArray(),
                output_names, expect_failure);

        void DefineT(int num_opers, Operation[] opers,
            TF_Output[] inputs, TF_Output[] outputs,
            string[] output_names, bool expect_failure = false)
        {
            func_ = c_api.TF_GraphToFunction(func_graph_, func_name_, false,
                num_opers, num_opers == -1 ? null : opers.Select(x => (IntPtr)x).ToArray(),
                inputs.Length, inputs.ToArray(),
                outputs.Length, outputs.ToArray(),
                output_names == null || output_names.Length == 0 ? null : output_names,
                IntPtr.Zero, null, s_.Handle);

            if (expect_failure)
            {
                ASSERT_EQ(IntPtr.Zero, func_);
                return;
            }

            ASSERT_EQ(TF_OK, s_.Code, s_.Message);
            ASSERT_NE(func_, IntPtr.Zero);
            ASSERT_EQ(func_name_, c_api.StringPiece(c_api.TF_FunctionName(func_)));
            c_api.TF_GraphCopyFunction(host_graph_, func_, IntPtr.Zero, s_.Handle);
            ASSERT_EQ(TF_OK, s_.Code, s_.Message);
        }

        Operation Use(Operation[] inputs)
            => UseT(inputs.Select(x => new TF_Output(x, 0)).ToArray());

        Operation UseT(TF_Output[] inputs)
            => UseHelper(inputs);

        Operation UseHelper(TF_Output[] inputs)
        {
            var desc = TF_NewOperation(host_graph_, func_name_, func_node_name_);
            foreach (var input in inputs)
                TF_AddInput(desc, input);
            c_api.TF_SetDevice(desc, "/cpu:0");
            var op = TF_FinishOperation(desc, s_);
            ASSERT_EQ(TF_OK, s_.Code, s_.Message);
            ASSERT_NE(op, IntPtr.Zero);

            return op;
        }

        void Run(KeyValuePair<Operation, Tensor>[] inputs, Operation output, int expected_result)
            => Run(inputs, new[] { new TF_Output(output, 0) }, new[] { expected_result });

        unsafe void Run(KeyValuePair<Operation, Tensor>[] inputs, TF_Output[] outputs, int[] expected_results)
        {
            var csession = new CSession(host_graph_, s_);
            ASSERT_EQ(TF_OK, s_.Code, s_.Message);

            csession.SetInputs(inputs);
            csession.SetOutputs(outputs);
            csession.Run(s_);
            ASSERT_EQ(TF_OK, s_.Code, s_.Message);

            for (int i = 0; i < expected_results.Length; ++i)
            {
                var output = csession.output_tensor(i);
                ASSERT_TRUE(output != IntPtr.Zero);
                EXPECT_EQ(TF_DataType.TF_INT32, c_api.TF_TensorType(output));
                EXPECT_EQ(0, c_api.TF_NumDims(output));
                ASSERT_EQ(sizeof(int), (int)c_api.TF_TensorByteSize(output));
                var output_contents = c_api.TF_TensorData(output);
                EXPECT_EQ(expected_results[i], *(int*)output_contents.ToPointer());
            }
        }

        void VerifyFDef(string[] nodes, List<IOSpec> inputs, List<IOSpec> outputs,
            List<EdgeSpec> e_edges, List<EdgeSpec> c_edges,
            bool is_exact_edges = true)
        {
            var fdef = GetFunctionDef(func_);
            EXPECT_NE(fdef, IntPtr.Zero);
            VerifyFDefNodes(fdef, nodes);
            VerifyFDefInputs(fdef, inputs);
            VerifyFDefOutputs(fdef, outputs);
            VerifyFDefEdges(fdef, e_edges, c_edges, is_exact_edges);
        }

        void VerifyFDefNodes(FunctionDef fdef, string[] nodes)
        {
            ASSERT_EQ(nodes.Length, fdef.NodeDef.Count);
            foreach (var node in fdef.NodeDef)
            {
                ASSERT_TRUE(nodes.Contains(node.Name), $"Got unexpected node: {node.Name} in fdef: {fdef}");
            }
        }

        void VerifyFDefInputs(FunctionDef fdef, List<IOSpec> inputs)
        {
            var signature = fdef.Signature;
            ASSERT_EQ(inputs.Count, signature.InputArg.Count);
            for (int i = 0; i < inputs.Count; ++i)
            {
                var arg = signature.InputArg[i];
                var input = inputs[i];
                if (input.Value != DataType.DtInvalid)
                    ASSERT_EQ(arg.Type, input.Value, $"");
                ASSERT_EQ(arg.Name, input.Key, $"Got unexpected name for input {i}. fdef: {fdef}");
            }
        }

        void VerifyFDefOutputs(FunctionDef fdef, List<IOSpec> outputs)
        {
            var signature = fdef.Signature;
            ASSERT_EQ(outputs.Count, signature.OutputArg.Count);
            for (int i = 0; i < outputs.Count; ++i)
            {
                var arg = signature.OutputArg[i];
                var output = outputs[i];
                if (output.Value != DataType.DtInvalid)
                    ASSERT_EQ(arg.Type, output.Value, $"");
                ASSERT_EQ(arg.Name, output.Key, $"Got unexpected name for input {i}. fdef: {fdef}");
            }
        }

        void VerifyFDefEdges(FunctionDef fdef, List<EdgeSpec> e_edges, List<EdgeSpec> c_edges, bool is_exact_edges = true)
        {
            // Build a set of edges from fdef
            var a_edges = new List<EdgeSpec>(); // actual edges
            // Get edges from inputs to body nodes and between body nodes
            foreach (var node in fdef.NodeDef)
            {
                for (int i = 0; i < node.Input.Count; ++i)
                {
                    var input = node.Input[i];
                    a_edges.Add(new EdgeSpec(input, $"{node.Name}:{i}"));
                }
            }
            // Get edges from body nodes to outputs and from inputs to outputs
            foreach (var arg in fdef.Signature.OutputArg)
            {
                var iter = fdef.Ret.FirstOrDefault(x => x.Key == arg.Name);
                if (iter.Key != null)
                {
                    a_edges.Add(new EdgeSpec(iter.Value, arg.Name));
                }
                else
                {
                    a_edges.Add(new EdgeSpec(arg.Name, arg.Name));
                }
            }
            // Verify edges
            foreach (var edge in e_edges)
            {
                ASSERT_TRUE(a_edges.Contains(edge));
            }
            foreach (var edge in c_edges)
            {
                ASSERT_TRUE(a_edges.Contains(edge));
            }
            // If caller specified all edges, check that we have seen all
            if (is_exact_edges)
            {
                ASSERT_EQ(e_edges.Count + c_edges.Count, a_edges.Count,
                    $"Expected edges: {e_edges}, Expected Control edges: {c_edges}, Actual edges: {a_edges}");
            }
        }

        public void Dispose()
        {

        }

        public struct IOSpec
        {
            KeyValuePair<string, DataType> pair;
            public string Key => pair.Key;
            public DataType Value => pair.Value;

            public IOSpec(string key, DataType value = DataType.DtInvalid)
            {
                pair = new KeyValuePair<string, DataType>(key, value);
            }
        }

        public struct EdgeSpec
        {
            KeyValuePair<string, string> pair;
            public string Key => pair.Key;
            public string Value => pair.Value;

            public EdgeSpec(string key, string value)
            {
                pair = new KeyValuePair<string, string>(key, value);
            }
        }
    }
}
