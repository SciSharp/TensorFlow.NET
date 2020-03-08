using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using Tensorflow;
using Tensorflow.Eager;
using Buffer = System.Buffer;

namespace TensorFlowNET.UnitTest.Eager
{
    /// <summary>
    /// tensorflow\c\eager\c_api_test.cc
    /// </summary>
    [TestClass]
    public partial class CApiEagerTest : CApiTest
    {
        IntPtr TestMatrixTensorHandle()
        {
            var dims = new long[] { 2, 2 };
            var data = new float[] { 1.0f, 2.0f, 3.0f, 4.0f };
            var t = c_api.TF_AllocateTensor(TF_FLOAT, dims, dims.Length, (ulong)data.Length * sizeof(float));
            memcpy(data, c_api.TF_TensorData(t), data.Length * sizeof(float));
            
            var status = c_api.TF_NewStatus();
            var th = c_api.TFE_NewTensorHandle(t, status);
            CHECK_EQ(TF_OK, TF_GetCode(status), TF_Message(status));
            c_api.TF_DeleteTensor(t);
            c_api.TF_DeleteStatus(status);
            return th;
        }

        IntPtr MatMulOp(IntPtr ctx, IntPtr a, IntPtr b)
        {
            var status = TF_NewStatus();

            var op = TFE_NewOp(ctx, "MatMul", status);
            CHECK_EQ(TF_OK, TF_GetCode(status), TF_Message(status));
            TFE_OpAddInput(op, a, status);
            CHECK_EQ(TF_OK, TF_GetCode(status), TF_Message(status));
            TFE_OpAddInput(op, b, status);
            CHECK_EQ(TF_OK, TF_GetCode(status), TF_Message(status));
            TF_DeleteStatus(status);
            TFE_OpSetAttrType(op, "T", TFE_TensorHandleDataType(a));

            return op;
        }
    }
}
