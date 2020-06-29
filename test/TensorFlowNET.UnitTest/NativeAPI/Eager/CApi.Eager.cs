﻿using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using Tensorflow;
using Tensorflow.Eager;
using static Tensorflow.Binding;

namespace TensorFlowNET.UnitTest.NativeAPI
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
            tf.memcpy(c_api.TF_TensorData(t), data, data.Length * sizeof(float));
            
            using var status = c_api.TF_NewStatus();
            var th = c_api.TFE_NewTensorHandle(t, status);
            CHECK_EQ(TF_OK, TF_GetCode(status), TF_Message(status));
            c_api.TF_DeleteTensor(t);
            return th;
        }

        IntPtr MatMulOp(SafeContextHandle ctx, IntPtr a, IntPtr b)
        {
            using var status = TF_NewStatus();

            var op = TFE_NewOp(ctx, "MatMul", status);
            CHECK_EQ(TF_OK, TF_GetCode(status), TF_Message(status));
            TFE_OpAddInput(op, a, status);
            CHECK_EQ(TF_OK, TF_GetCode(status), TF_Message(status));
            TFE_OpAddInput(op, b, status);
            CHECK_EQ(TF_OK, TF_GetCode(status), TF_Message(status));
            TFE_OpSetAttrType(op, "T", TFE_TensorHandleDataType(a));

            return op;
        }

        bool GetDeviceName(SafeContextHandle ctx, ref string device_name, string device_type)
        {
            using var status = TF_NewStatus();
            using var devices = TFE_ContextListDevices(ctx, status);
            CHECK_EQ(TF_OK, TF_GetCode(status), TF_Message(status));

            int num_devices = TF_DeviceListCount(devices);
            for (int i = 0; i < num_devices; ++i)
            {
                var dev_type = TF_DeviceListType(devices, i, status);
                CHECK_EQ(TF_GetCode(status), TF_OK, TF_Message(status));
                var dev_name = TF_DeviceListName(devices, i, status);
                CHECK_EQ(TF_GetCode(status), TF_OK, TF_Message(status));
                if (dev_type == device_type)
                {
                    device_name = dev_name;
                    return true;
                }
            }

            return false;
        }

        IntPtr ShapeOp(SafeContextHandle ctx, IntPtr a)
        {
            using var status = TF_NewStatus();

            var op = TFE_NewOp(ctx, "Shape", status);
            CHECK_EQ(TF_OK, TF_GetCode(status), TF_Message(status));
            TFE_OpAddInput(op, a, status);
            CHECK_EQ(TF_OK, TF_GetCode(status), TF_Message(status));
            TFE_OpSetAttrType(op, "T", TFE_TensorHandleDataType(a));

            return op;
        }

        unsafe IntPtr CreateVariable(SafeContextHandle ctx, float value, SafeStatusHandle status)
        {
            var op = TFE_NewOp(ctx, "VarHandleOp", status);
            if (TF_GetCode(status) != TF_OK) return IntPtr.Zero;
            TFE_OpSetAttrType(op, "dtype", TF_FLOAT);
            TFE_OpSetAttrShape(op, "shape", new long[0], 0, status);
            TFE_OpSetAttrString(op, "container", "", 0);
            TFE_OpSetAttrString(op, "shared_name", "", 0);
            if (TF_GetCode(status) != TF_OK) return IntPtr.Zero;
            var var_handle = new IntPtr[1];
            int num_retvals = 1;
            TFE_Execute(op, var_handle, ref num_retvals, status);
            TFE_DeleteOp(op);
            if (TF_GetCode(status) != TF_OK) return IntPtr.Zero;
            CHECK_EQ(1, num_retvals);

            // Assign 'value' to it.
            op = TFE_NewOp(ctx, "AssignVariableOp", status);
            if (TF_GetCode(status) != TF_OK) return IntPtr.Zero;
            TFE_OpSetAttrType(op, "dtype", TF_FLOAT);
            TFE_OpAddInput(op, var_handle[0], status);

            // Convert 'value' to a TF_Tensor then a TFE_TensorHandle.
            var t = c_api.TF_AllocateTensor(TF_DataType.TF_FLOAT, new long[0], 0, sizeof(float));
            tf.memcpy(TF_TensorData(t).ToPointer(), &value, TF_TensorByteSize(t));

            var value_handle = c_api.TFE_NewTensorHandle(t, status);
            if (TF_GetCode(status) != TF_OK) return IntPtr.Zero;

            TFE_OpAddInput(op, value_handle, status);
            if (TF_GetCode(status) != TF_OK) return IntPtr.Zero;

            num_retvals = 0;
            c_api.TFE_Execute(op, null, ref num_retvals, status);
            TFE_DeleteOp(op);
            if (TF_GetCode(status) != TF_OK) return IntPtr.Zero;
            CHECK_EQ(0, num_retvals);

            return var_handle[0];
        }

        IntPtr TestAxisTensorHandle()
        {
            var dims = new long[] { 1 };
            var data = new int[] { 1 };
            var t = c_api.TF_AllocateTensor(TF_DataType.TF_INT32, dims, 1, sizeof(int));
            tf.memcpy(TF_TensorData(t), data, TF_TensorByteSize(t));
            using var status = TF_NewStatus();
            var th = c_api.TFE_NewTensorHandle(t, status);
            CHECK_EQ(TF_OK, TF_GetCode(status), TF_Message(status));
            TF_DeleteTensor(t);
            return th;
        }

        IntPtr TestScalarTensorHandle(bool value)
        {
            var data = new[] { value };
            var t = c_api.TF_AllocateTensor(TF_BOOL, null, 0, sizeof(bool));
            tf.memcpy(TF_TensorData(t), data, TF_TensorByteSize(t));
            using var status = TF_NewStatus();
            var th = TFE_NewTensorHandle(t, status);
            CHECK_EQ(TF_OK, TF_GetCode(status), TF_Message(status));
            TF_DeleteTensor(t);
            return th;
        }

        IntPtr TestScalarTensorHandle(float value)
        {
            var data = new [] { value };
            var t = c_api.TF_AllocateTensor(TF_FLOAT, null, 0, sizeof(float));
            tf.memcpy(TF_TensorData(t), data, TF_TensorByteSize(t));
            using var status = TF_NewStatus();
            var th = TFE_NewTensorHandle(t, status);
            CHECK_EQ(TF_OK, TF_GetCode(status), TF_Message(status));
            TF_DeleteTensor(t);
            return th;
        }
    }
}
