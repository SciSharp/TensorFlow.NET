using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using Tensorflow.Device;
using Tensorflow.Eager;

namespace Tensorflow.Native.UnitTest
{
    public class CApiTest
    {
        protected static readonly TF_Code TF_OK = TF_Code.TF_OK;
        protected static readonly TF_DataType TF_FLOAT = TF_DataType.TF_FLOAT;
        protected static readonly TF_DataType TF_BOOL = TF_DataType.TF_BOOL;

        protected void EXPECT_TRUE(bool expected, string msg = "")
            => Assert.IsTrue(expected, msg);

        protected static void EXPECT_EQ(object expected, object actual, string msg = "")
            => Assert.AreEqual(expected, actual, msg);

        protected void CHECK_EQ(object expected, object actual, string msg = "")
            => Assert.AreEqual(expected, actual, msg);

        protected void EXPECT_NE(object expected, object actual, string msg = "")
            => Assert.AreNotEqual(expected, actual, msg);

        protected void CHECK_NE(object expected, object actual, string msg = "")
            => Assert.AreNotEqual(expected, actual, msg);

        protected void EXPECT_GE(int expected, int actual, string msg = "")
            => Assert.IsTrue(expected >= actual, msg);

        protected void ASSERT_EQ(object expected, object actual, string msg = "")
            => Assert.AreEqual(expected, actual, msg);

        protected void ASSERT_NE(object expected, object actual, string msg = "")
            => Assert.AreNotEqual(expected, actual, msg);

        protected void ASSERT_TRUE(bool condition, string msg = "")
            => Assert.IsTrue(condition, msg);

        protected OperationDescription TF_NewOperation(Graph graph, string opType, string opName)
            => c_api.TF_NewOperation(graph, opType, opName);

        protected void TF_AddInput(OperationDescription desc, TF_Output input)
            => c_api.TF_AddInput(desc, input);

        protected Operation TF_FinishOperation(OperationDescription desc, Status s)
            => c_api.TF_FinishOperation(desc, s.Handle);

        protected void TF_SetAttrTensor(OperationDescription desc, string attrName, Tensor value, Status s)
            => c_api.TF_SetAttrTensor(desc, attrName, value, s.Handle);

        protected void TF_SetAttrType(OperationDescription desc, string attrName, TF_DataType dtype)
            => c_api.TF_SetAttrType(desc, attrName, dtype);

        protected void TF_SetAttrBool(OperationDescription desc, string attrName, bool value)
            => c_api.TF_SetAttrBool(desc, attrName, value);

        protected TF_DataType TFE_TensorHandleDataType(SafeTensorHandleHandle h)
            => c_api.TFE_TensorHandleDataType(h);

        protected int TFE_TensorHandleNumDims(SafeTensorHandleHandle h, SafeStatusHandle status)
            => c_api.TFE_TensorHandleNumDims(h, status);

        protected TF_Code TF_GetCode(Status s)
            => s.Code;

        protected static TF_Code TF_GetCode(SafeStatusHandle s)
            => c_api.TF_GetCode(s);

        protected static string TF_Message(SafeStatusHandle s)
            => c_api.StringPiece(c_api.TF_Message(s));

        protected SafeStatusHandle TF_NewStatus()
            => c_api.TF_NewStatus();

        protected void TF_DeleteTensor(IntPtr t)
            => c_api.TF_DeleteTensor(t);

        protected IntPtr TF_TensorData(IntPtr t)
            => c_api.TF_TensorData(t);

        protected ulong TF_TensorByteSize(IntPtr t)
            => c_api.TF_TensorByteSize(t);

        protected void TFE_OpAddInput(SafeOpHandle op, SafeTensorHandleHandle h, SafeStatusHandle status)
            => c_api.TFE_OpAddInput(op, h, status);

        protected void TFE_OpSetAttrType(SafeOpHandle op, string attr_name, TF_DataType value)
            => c_api.TFE_OpSetAttrType(op, attr_name, value);

        protected void TFE_OpSetAttrShape(SafeOpHandle op, string attr_name, long[] dims, int num_dims, SafeStatusHandle out_status)
            => c_api.TFE_OpSetAttrShape(op, attr_name, dims, num_dims, out_status);

        protected void TFE_OpSetAttrString(SafeOpHandle op, string attr_name, string value, uint length)
            => c_api.TFE_OpSetAttrString(op, attr_name, value, length);

        protected SafeOpHandle TFE_NewOp(SafeContextHandle ctx, string op_or_function_name, SafeStatusHandle status)
            => c_api.TFE_NewOp(ctx, op_or_function_name, status);

        protected SafeTensorHandleHandle TFE_NewTensorHandle(IntPtr t, SafeStatusHandle status)
            => c_api.TFE_NewTensorHandle(t, status);

        protected void TFE_Execute(SafeOpHandle op, SafeTensorHandleHandle[] retvals, out int num_retvals, SafeStatusHandle status)
            => c_api.TFE_Execute(op, retvals, out num_retvals, status);

        protected SafeContextOptionsHandle TFE_NewContextOptions()
            => c_api.TFE_NewContextOptions();

        protected SafeContextHandle TFE_NewContext(SafeContextOptionsHandle opts, SafeStatusHandle status)
            => c_api.TFE_NewContext(opts, status);

        protected int TFE_OpGetInputLength(SafeOpHandle op, string input_name, SafeStatusHandle status)
            => c_api.TFE_OpGetInputLength(op, input_name, status);

        protected int TFE_OpAddInputList(SafeOpHandle op, SafeTensorHandleHandle[] inputs, int num_inputs, SafeStatusHandle status)
            => c_api.TFE_OpAddInputList(op, inputs, num_inputs, status);

        protected int TFE_OpGetOutputLength(SafeOpHandle op, string input_name, SafeStatusHandle status)
            => c_api.TFE_OpGetOutputLength(op, input_name, status);

        protected void TFE_DeleteTensorHandle(IntPtr h)
            => c_api.TFE_DeleteTensorHandle(h);

        protected SafeExecutorHandle TFE_ContextGetExecutorForThread(SafeContextHandle ctx)
            => c_api.TFE_ContextGetExecutorForThread(ctx);

        protected void TFE_ExecutorWaitForAllPendingNodes(SafeExecutorHandle executor, SafeStatusHandle status)
            => c_api.TFE_ExecutorWaitForAllPendingNodes(executor, status);

        protected IntPtr TFE_TensorHandleResolve(SafeTensorHandleHandle h, SafeStatusHandle status)
            => c_api.TFE_TensorHandleResolve(h, status);

        protected string TFE_TensorHandleDeviceName(SafeTensorHandleHandle h, SafeStatusHandle status)
            => c_api.StringPiece(c_api.TFE_TensorHandleDeviceName(h, status));

        protected string TFE_TensorHandleBackingDeviceName(SafeTensorHandleHandle h, SafeStatusHandle status)
            => c_api.StringPiece(c_api.TFE_TensorHandleBackingDeviceName(h, status));

        protected SafeDeviceListHandle TFE_ContextListDevices(SafeContextHandle ctx, SafeStatusHandle status)
            => c_api.TFE_ContextListDevices(ctx, status);

        protected int TF_DeviceListCount(SafeDeviceListHandle list)
            => c_api.TF_DeviceListCount(list);

        protected string TF_DeviceListType(SafeDeviceListHandle list, int index, SafeStatusHandle status)
            => c_api.StringPiece(c_api.TF_DeviceListType(list, index, status));

        protected string TF_DeviceListName(SafeDeviceListHandle list, int index, SafeStatusHandle status)
            => c_api.TF_DeviceListName(list, index, status);

        protected SafeTensorHandleHandle TFE_TensorHandleCopyToDevice(SafeTensorHandleHandle h, SafeContextHandle ctx, string device_name, SafeStatusHandle status)
            => c_api.TFE_TensorHandleCopyToDevice(h, ctx, device_name, status);

        protected void TFE_OpSetDevice(SafeOpHandle op, string device_name, SafeStatusHandle status)
            => c_api.TFE_OpSetDevice(op, device_name, status);
    }
}
