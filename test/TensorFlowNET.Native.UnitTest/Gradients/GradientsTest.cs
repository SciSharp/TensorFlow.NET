using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using System;
using Tensorflow.Util;

namespace Tensorflow.Native.UnitTest
{
    /// <summary>
    /// tensorflow\c\c_api_test.cc
    /// `class CApiGradientsTest`
    /// </summary>
    [TestClass]
    public class GradientsTest : CApiTest, IDisposable
    {
        private Graph graph_ = new Graph();
        private Graph expected_graph_ = new Graph();
        private Status s_ = new Status();

        private void TestGradientsSuccess(bool grad_inputs_provided)
        {
            var inputs = new TF_Output[2];
            var outputs = new TF_Output[1];
            var grad_outputs = new TF_Output[2];
            var expected_grad_outputs = new TF_Output[2];

            BuildSuccessGraph(inputs, outputs);
            BuildExpectedGraph(grad_inputs_provided, expected_grad_outputs);

            AddGradients(grad_inputs_provided, "gradients", inputs, 2, outputs, 1,
                 grad_outputs);
            EXPECT_EQ(TF_OK, TF_GetCode(s_));

            // Compare that the graphs match.
            GraphDef expected_gdef;
            GraphDef gdef;
            EXPECT_TRUE(GetGraphDef(expected_graph_, out expected_gdef));
            EXPECT_TRUE(GetGraphDef(graph_, out gdef));
            // Assert.IsTrue(expected_gdef.ToString().Equals(gdef.ToString()));

            // Compare that the output of the gradients of both graphs match.
            RunGraphsAndCompareOutputs(grad_outputs, expected_grad_outputs);
        }

        private bool GetGraphDef(Graph graph, out GraphDef graph_def)
        {
            graph_def = null;
            using (var s = new Status())
            {
                using (var buffer = new Buffer())
                {
                    c_api.TF_GraphToGraphDef(graph, buffer.Handle, s.Handle);
                    bool ret = TF_GetCode(s) == TF_OK;
                    EXPECT_EQ(TF_OK, TF_GetCode(s));
                    if (ret)
                        graph_def = GraphDef.Parser.ParseFrom(buffer.DangerousMemoryBlock.Stream());
                    return ret;
                }
            }
        }

        private void RunGraphsAndCompareOutputs(TF_Output[] grad_outputs, TF_Output[] expected_grad_outputs)
        {
            var csession = new CSession(graph_, s_);
            var expected_csession = new CSession(expected_graph_, s_);

            var grad_outputs_vec = grad_outputs;
            csession.SetOutputs(grad_outputs_vec);
            csession.Run(s_);
            ASSERT_EQ(TF_OK, TF_GetCode(s_));
            var out0 = csession.output_tensor(0);
            var out1 = csession.output_tensor(1);

            var expected_grad_outputs_vec = expected_grad_outputs;
            expected_csession.SetOutputs(expected_grad_outputs_vec);
            expected_csession.Run(s_);
            ASSERT_EQ(TF_OK, TF_GetCode(s_));
            var expected_out0 = expected_csession.output_tensor(0);
            var expected_out1 = expected_csession.output_tensor(1);

            //CompareTensors(out0, expected_out0);
            //CompareTensors(out1, expected_out1);
        }
        /*void TestGradientsError(bool grad_inputs_provided)
        {
            var inputs = new TF_Output[1];
            var outputs = new TF_Output[1];
            var grad_outputs = new TF_Output[1];

            BuildErrorGraph(inputs, outputs);

            AddGradients(grad_inputs_provided, nullptr, inputs, 1, outputs, 1,
                         grad_outputs);

            string expected_msg =
                "No gradient defined for op: TestOpWithNoGradient. Please see "
        "https://www.tensorflow.org/code/"
        "tensorflow/cc/gradients/README.md"
        " for instructions on how to add C++ gradients.";
            EXPECT_EQ(expected_msg, TF_Message(s_));
        }*/

        private void AddGradients(bool grad_inputs_provided, string prefix, TF_Output[] inputs, int ninputs,
            TF_Output[] outputs, int noutputs, TF_Output[] grad_outputs)
        {
            if (grad_inputs_provided)
            {
                var grad_inputs = new TF_Output[1];
                float[] grad_inputs_val = { 1.0f, 1.0f, 1.0f, 1.0f };
                var grad_inputs_op = FloatConst2x2(graph_, s_, grad_inputs_val, "GradInputs");
                grad_inputs[0] = new TF_Output(grad_inputs_op, 0);

                IntPtr[] handles = new IntPtr[2] { IntPtr.Zero, IntPtr.Zero };
                c_api.TF_AddGradientsWithPrefix(graph_, prefix, outputs, noutputs, inputs,
                                      ninputs, grad_inputs, s_.Handle, handles);

                var op = new Operation(handles[0]);
            }
            else
            {
                //c_api.TF_AddGradientsWithPrefix(graph_, prefix, outputs, noutputs, inputs,
                //ninputs, null, s_, grad_outputs);
            }
        }

        private void BuildSuccessGraph(TF_Output[] inputs, TF_Output[] outputs)
        {
            // Construct the following graph:
            //            |
            //           z|
            //            |
            //          MatMul
            //         /       \
            //        ^         ^
            //        |         |
            //       x|        y|
            //        |         |
            //        |         |
            //      Const_0    Const_1
            //
            var const0_val = new float[] { 1.0f, 2.0f, 3.0f, 4.0f };
            var const1_val = new float[] { 1.0f, 0.0f, 0.0f, 1.0f };
            var const0 = FloatConst2x2(graph_, s_, const0_val, "Const_0");
            var const1 = FloatConst2x2(graph_, s_, const1_val, "Const_1");
            var matmul = MatMul(graph_, s_, const0, const1, "MatMul");
            inputs[0] = new TF_Output(const0, 0);
            inputs[1] = new TF_Output(const1, 0);
            outputs[0] = new TF_Output(matmul, 0);
            EXPECT_EQ(TF_OK, TF_GetCode(s_));
        }

        private void BuildExpectedGraph(bool grad_inputs_provided, TF_Output[] expected_grad_outputs)
        {
            // The expected graph looks like this if grad_inputs_provided.
            // If grad_inputs_provided is false, Const_0 will be a OnesLike op.
            //      ^             ^
            //    dy|           dx|        // MatMul Gradient Graph
            //      |             |
            //   MatMul_2      MatMul_1
            //   ^   ^          ^    ^
            //   |   |----------|    |
            //   |        ^          |
            //   |      dz|          |
            //   |        |          |
            //   |     Const_3       |
            //   |                   |
            //   |        ^          |
            //   |       z|          |     // MatMul Forward Graph
            //   |        |          |
            //   |      MatMul       |
            //   |     /       \     |
            //   |    ^         ^    |
            //   |    |         |    |
            //   |---x|        y|----|
            //        |         |
            //        |         |
            //      Const_0   Const_1
            //
            float[] const0_val = { 1.0f, 2.0f, 3.0f, 4.0f };
            float[] const1_val = { 1.0f, 0.0f, 0.0f, 1.0f };
            var const0 = FloatConst2x2(expected_graph_, s_, const0_val, "Const_0");
            var const1 = FloatConst2x2(expected_graph_, s_, const1_val, "Const_1");
            var matmul = MatMul(expected_graph_, s_, const0, const1, "MatMul");

            Operation const3;
            if (grad_inputs_provided)
            {
                float[] const3_val = { 1.0f, 1.0f, 1.0f, 1.0f };
                const3 = FloatConst2x2(expected_graph_, s_, const3_val, "GradInputs");
            }
            else
            {
                const3 = OnesLike(expected_graph_, s_, matmul, "gradients/OnesLike");
            }

            var matmul1 = MatMul(expected_graph_, s_, const3, const1,
                                   "gradients/MatMul", false, true);
            var matmul2 = MatMul(expected_graph_, s_, const0, const3,
                                           "gradients/MatMul_1", true, false);
            expected_grad_outputs[0] = new TF_Output(matmul1, 0);
            expected_grad_outputs[1] = new TF_Output(matmul2, 0);
        }

        private Operation OnesLike(Graph graph, Status s, Operation input, string name)
        {
            var desc = TF_NewOperation(graph, "OnesLike", name);
            TF_AddInput(desc, new TF_Output(input, 0));
            var op = TF_FinishOperation(desc, s);
            EXPECT_EQ(TF_OK, TF_GetCode(s));
            return op;
        }

        private Operation FloatConst2x2(Graph graph, Status s, float[] values, string name)
        {
            var tensor = FloatTensor2x2(values);
            var desc = TF_NewOperation(graph, "Const", name);
            TF_SetAttrTensor(desc, "value", tensor, s);
            if (TF_GetCode(s) != TF_OK) return IntPtr.Zero;
            TF_SetAttrType(desc, "dtype", TF_FLOAT);
            var op = TF_FinishOperation(desc, s);
            EXPECT_EQ(TF_OK, TF_GetCode(s));
            return op;
        }

        private Tensor FloatTensor2x2(float[] values)
        {
            //long[] dims = { 2, 2 };
            //Tensor t = c_api.TF_AllocateTensor(TF_FLOAT, dims, 2, sizeof(float) * 4);
            //Marshal.Copy(values, 0, t, 4);
            Tensor t = new Tensor(new NDArray(values).reshape(2, 2));
            return t;
        }

        private Operation MatMul(Graph graph, Status s, Operation l, Operation r, string name,
            bool transpose_a = false, bool transpose_b = false)
        {
            var desc = TF_NewOperation(graph, "MatMul", name);
            if (transpose_a)
            {
                TF_SetAttrBool(desc, "transpose_a", true);
            }
            if (transpose_b)
            {
                TF_SetAttrBool(desc, "transpose_b", true);
            }
            TF_AddInput(desc, new TF_Output(l, 0));
            TF_AddInput(desc, new TF_Output(r, 0));
            var op = TF_FinishOperation(desc, s);
            EXPECT_EQ(TF_OK, TF_GetCode(s));
            return op;
        }

        [TestMethod]
        public void Gradients_GradInputs()
        {
            //TestGradientsSuccess(true);
        }

        [TestMethod]
        public void Gradients_NoGradInputs()
        {
            //TestGradientsSuccess(false);
        }

        [TestMethod]
        public void OpWithNoGradientRegistered_GradInputs()
        {
            //TestGradientsError(true);
        }

        [TestMethod]
        public void OpWithNoGradientRegistered_NoGradInputs()
        {
            //TestGradientsError(false);
        }

        public void Dispose()
        {
            graph_.Dispose();
            expected_graph_.Dispose();
            s_.Dispose();
        }
    }
}
