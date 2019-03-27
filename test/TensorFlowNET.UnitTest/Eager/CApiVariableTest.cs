using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow;
using Tensorflow.Eager;

namespace TensorFlowNET.UnitTest.Eager
{
    /// <summary>
    /// tensorflow\c\eager\c_api_test.cc
    /// </summary>
    [TestClass]
    public class CApiVariableTest : CApiTest, IDisposable
    {
        Status status = new Status();
        ContextOptions opts = new ContextOptions();
        Context ctx;

        //[TestMethod]
        public void Variables()
        {
            ctx = new Context(opts, status);
            ASSERT_EQ(TF_Code.TF_OK, status.Code);
            opts.Dispose();

            var var_handle = CreateVariable(ctx, 12.0F);
            ASSERT_EQ(TF_OK, TF_GetCode(status));
        }

        private IntPtr CreateVariable(Context ctx, float value)
        {
            // Create the variable handle.
            var op = c_api.TFE_NewOp(ctx, "VarHandleOp", status);
            if (TF_GetCode(status) != TF_OK) return IntPtr.Zero;

            c_api.TFE_OpSetAttrType(op, "dtype", TF_DataType.TF_FLOAT);
            c_api.TFE_OpSetAttrShape(op, "shape", new long[0], 0, status);
            c_api.TFE_OpSetAttrString(op, "container", "", 0);
            c_api.TFE_OpSetAttrString(op, "shared_name", "", 0);
            if (TF_GetCode(status) != TF_OK) return IntPtr.Zero;
            var var_handle = IntPtr.Zero;
            int[] num_retvals = { 1 };
            c_api.TFE_Execute(op, var_handle, num_retvals, status);
            c_api.TFE_DeleteOp(op);
            if (TF_GetCode(status) != TF_OK) return IntPtr.Zero;
            ASSERT_EQ(1, num_retvals);

            // Assign 'value' to it.
            op = c_api.TFE_NewOp(ctx, "AssignVariableOp", status);
            if (TF_GetCode(status) != TF_OK) return IntPtr.Zero;
            c_api.TFE_OpSetAttrType(op, "dtype", TF_FLOAT);
            c_api.TFE_OpAddInput(op, var_handle, status);

            // Convert 'value' to a TF_Tensor then a TFE_TensorHandle.
            var t = new Tensor(value);

            var value_handle = c_api.TFE_NewTensorHandle(t);
            if (TF_GetCode(status) != TF_OK) return IntPtr.Zero;

            c_api.TFE_OpAddInput(op, value_handle, status);
            if (TF_GetCode(status) != TF_OK) return IntPtr.Zero;

            num_retvals = new int[] { 0 };
            c_api.TFE_Execute(op, IntPtr.Zero, num_retvals, status);
            c_api.TFE_DeleteOp(op);
            if (TF_GetCode(status) != TF_OK) return IntPtr.Zero;
            ASSERT_EQ(0, num_retvals);

            return var_handle;
        }

        public void Dispose()
        {
            status.Dispose();
            opts.Dispose();
            ctx.Dispose();
        }
    }
}
