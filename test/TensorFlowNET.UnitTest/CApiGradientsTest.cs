using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Core;
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;
using Tensorflow;

namespace TensorFlowNET.UnitTest
{
    /// <summary>
    /// tensorflow\c\c_api_test.cc
    /// `class CApiGradientsTest`
    /// </summary>
    [TestClass]
    public class CApiGradientsTest : CApiTest, IDisposable
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
            long[] dims = { 2, 2 };
            Tensor t = c_api.TF_AllocateTensor(TF_FLOAT, dims, 2, sizeof(float) * 4);
            Marshal.Copy(values, 0, t, 4);
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
            TestGradientsSuccess(true);
        }

        [TestMethod]
        public void Gradients_NoGradInputs()
        {
            TestGradientsSuccess(false);
        }

        public void Dispose()
        {
            graph_.Dispose();
            expected_graph_.Dispose();
            s_.Dispose();
        }
    }
}
