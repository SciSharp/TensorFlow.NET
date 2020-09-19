using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using Tensorflow;
using Tensorflow.Functions;
using static TensorFlowNET.UnitTest.c_test_util;

namespace TensorFlowNET.UnitTest.NativeAPI
{
    /// <summary>
    /// tensorflow\c\c_api_function_test.cc
    /// `class CApiColocationTest`
    /// </summary>
    [TestClass]
    public class CApiFunctionTest : CApiTest, IDisposable
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
            VerifyFDef(new[] { "scalar10_0" });
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
            IntPtr output_names_ptr = IntPtr.Zero;

            func_ = c_api.TF_GraphToFunction(func_graph_, func_name_, false,
                num_opers, num_opers == -1 ? new IntPtr[0] : opers.Select(x => (IntPtr)x).ToArray(),
                inputs.Length, inputs.ToArray(),
                outputs.Length, outputs.ToArray(),
                output_names_ptr, IntPtr.Zero, null, s_.Handle);

            // delete output_names_ptr

            if (expect_failure)
            {
                ASSERT_EQ(IntPtr.Zero, func_);
                return;
            }

            ASSERT_EQ(TF_OK, s_.Code, s_.Message);
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
                ASSERT_NE(output, IntPtr.Zero);
                EXPECT_EQ(TF_DataType.TF_INT32, c_api.TF_TensorType(output));
                EXPECT_EQ(0, c_api.TF_NumDims(output));
                ASSERT_EQ(sizeof(int), (int)c_api.TF_TensorByteSize(output));
                var output_contents = c_api.TF_TensorData(output);
                EXPECT_EQ(expected_results[i], *(int*)output_contents.ToPointer());
            }
        }

        void VerifyFDef(string[] nodes)
        {
            var fdef = GetFunctionDef(func_);
            EXPECT_NE(fdef, IntPtr.Zero);
            VerifyFDefNodes(fdef, nodes);
        }

        void VerifyFDefNodes(FunctionDef fdef, string[] nodes)
        {
            ASSERT_EQ(nodes.Length, fdef.NodeDef.Count);
        }

        public void Dispose()
        {
            
        }
    }
}
