using Google.Protobuf;
using System;
using System.Runtime.InteropServices;
using Tensorflow.Device;
using Tensorflow.Eager;
using Tensorflow.Util;

namespace Tensorflow
{
    public partial class c_api
    {
        /// <summary>
        /// Return a new options object.
        /// </summary>
        /// <returns>TFE_ContextOptions*</returns>
        [DllImport(TensorFlowLibName)]
        public static extern SafeContextOptionsHandle TFE_NewContextOptions();

        /// <summary>
        /// Destroy an options object.
        /// </summary>
        /// <param name="options">TFE_ContextOptions*</param>
        [DllImport(TensorFlowLibName)]
        public static extern void TFE_DeleteContextOptions(IntPtr options);

        /// <summary>
        /// 
        /// </summary>
        /// <param name="op">TFE_Op*</param>
        /// <param name="attr_name">const char*</param>
        /// <param name="is_list">unsigned char*</param>
        /// <param name="status">TF_Status*</param>
        /// <returns></returns>
        [DllImport(TensorFlowLibName)]
        public static extern TF_AttrType TFE_OpGetAttrType(SafeOpHandle op, string attr_name, ref byte is_list, SafeStatusHandle status);

        [DllImport(TensorFlowLibName)]
        public static extern TF_AttrType TFE_OpNameGetAttrType(SafeContextHandle ctx, string op_or_function_name, string attr_name, ref byte is_list, SafeStatusHandle status);

        /// <summary>
        /// Returns the length (number of tensors) of the input argument `input_name`
        /// found in the provided `op`.
        /// </summary>
        /// <param name="op">TFE_Op*</param>
        /// <param name="input_name">const char*</param>
        /// <param name="status">TF_Status*</param>
        [DllImport(TensorFlowLibName)]
        public static extern int TFE_OpGetInputLength(SafeOpHandle op, string input_name, SafeStatusHandle status);

        /// <summary>
        /// Returns the length (number of tensors) of the output argument `output_name`
        /// found in the provided `op`.
        /// </summary>
        /// <param name="op"></param>
        /// <param name="input_name"></param>
        /// <param name="status"></param>
        /// <returns></returns>
        [DllImport(TensorFlowLibName)]
        public static extern int TFE_OpGetOutputLength(SafeOpHandle op, string input_name, SafeStatusHandle status);

        /// <summary>
        /// 
        /// </summary>
        /// <param name="op">TFE_Op*</param>
        /// <param name="inputs">TFE_TensorHandle**</param>
        /// <param name="num_inputs">int</param>
        /// <param name="status">TF_Status*</param>
        /// <returns></returns>
        [DllImport(TensorFlowLibName)]
        public static extern int TFE_OpAddInputList(SafeOpHandle op, [In, MarshalAs(UnmanagedType.CustomMarshaler, MarshalTypeRef = typeof(SafeHandleArrayMarshaler))] SafeTensorHandleHandle[] inputs, int num_inputs, SafeStatusHandle status);

        /// <summary>
        /// 
        /// </summary>
        /// <param name="opts">const TFE_ContextOptions*</param>
        /// <param name="status">TF_Status*</param>
        /// <returns>TFE_Context*</returns>
        [DllImport(TensorFlowLibName)]
        public static extern SafeContextHandle TFE_NewContext(SafeContextOptionsHandle opts, SafeStatusHandle status);

        [DllImport(TensorFlowLibName)]
        public static extern void TFE_ContextStartStep(SafeContextHandle ctx);

        [DllImport(TensorFlowLibName)]
        public static extern void TFE_ContextEndStep(SafeContextHandle ctx);

        /// <summary>
        /// 
        /// </summary>
        /// <param name="ctx">TFE_Context*</param>
        [DllImport(TensorFlowLibName)]
        public static extern void TFE_DeleteContext(IntPtr ctx);

        public static void TFE_Execute(SafeOpHandle op, SafeTensorHandleHandle[] retvals, out int num_retvals, SafeStatusHandle status)
        {
            unsafe
            {
                num_retvals = retvals?.Length ?? 0;
                var rawReturns = stackalloc IntPtr[num_retvals];
                TFE_Execute(op, rawReturns, ref num_retvals, status);
                for (var i = 0; i < num_retvals; i++)
                {
                    retvals[i] = new SafeTensorHandleHandle(rawReturns[i]);
                }
            }
        }

        /// <summary>
        /// Execute the operation defined by 'op' and return handles to computed
        /// tensors in `retvals`.
        /// </summary>
        /// <param name="op">TFE_Op*</param>
        /// <param name="retvals">TFE_TensorHandle**</param>
        /// <param name="num_retvals">int*</param>
        /// <param name="status">TF_Status*</param>
        [DllImport(TensorFlowLibName)]
        private static unsafe extern void TFE_Execute(SafeOpHandle op, IntPtr* retvals, ref int num_retvals, SafeStatusHandle status);

        /// <summary>
        /// 
        /// </summary>
        /// <param name="ctx">TFE_Context*</param>
        /// <param name="op_or_function_name">const char*</param>
        /// <param name="status">TF_Status*</param>
        /// <returns></returns>
        [DllImport(TensorFlowLibName)]
        public static extern SafeOpHandle TFE_NewOp(SafeContextHandle ctx, string op_or_function_name, SafeStatusHandle status);

        /// <summary>
        /// Resets `op_to_reset` with `op_or_function_name` and `raw_device_name`. This
        /// is for performance optimization by reusing an exiting unused op rather than
        /// creating a new op every time. If `raw_device_name` is `NULL` or empty, it
        /// does not set the device name. If it's not `NULL`, then it attempts to parse
        /// and set the device name. It's effectively `TFE_OpSetDevice`, but it is faster
        /// than separately calling it because if the existing op has the same
        /// `raw_device_name`, it skips parsing and just leave as it is.
        /// </summary>
        /// <param name="op_to_reset">TFE_Op*</param>
        /// <param name="op_or_function_name">const char*</param>
        /// <param name="raw_device_name">const char*</param>
        /// <param name="status">TF_Status*</param>
        [DllImport(TensorFlowLibName)]
        public static extern void TFE_OpReset(SafeOpHandle op_to_reset, string op_or_function_name, string raw_device_name, SafeStatusHandle status);

        /// <summary>
        /// 
        /// </summary>
        /// <param name="op">TFE_Op*</param>
        [DllImport(TensorFlowLibName)]
        public static extern void TFE_DeleteOp(IntPtr op);

        /// <summary>
        /// 
        /// </summary>
        /// <param name="op">TFE_Op*</param>
        /// <param name="attr_name">const char*</param>
        /// <param name="value">TF_DataType</param>
        [DllImport(TensorFlowLibName)]
        public static extern void TFE_OpSetAttrType(SafeOpHandle op, string attr_name, TF_DataType value);

        [DllImport(TensorFlowLibName)]
        public static extern void TFE_OpSetAttrInt(SafeOpHandle op, string attr_name, long value);

        /// <summary>
        /// 
        /// </summary>
        /// <param name="op">TFE_Op*</param>
        /// <param name="attr_name">const char*</param>
        /// <param name="dims">const int64_t*</param>
        /// <param name="num_dims">const int</param>
        /// <param name="out_status">TF_Status*</param>
        [DllImport(TensorFlowLibName)]
        public static extern void TFE_OpSetAttrShape(SafeOpHandle op, string attr_name, long[] dims, int num_dims, SafeStatusHandle out_status);

        [DllImport(TensorFlowLibName)]
        public static extern void TFE_OpSetAttrShapeList(SafeOpHandle op, string attr_name, IntPtr[] dims, int[] num_dims, int num_values, SafeStatusHandle out_status);

        [DllImport(TensorFlowLibName)]
        public static extern void TFE_OpSetAttrStringList(SafeOpHandle op, string attr_name, IntPtr[] values, int[] lengths, int num_values);

        [DllImport(TensorFlowLibName)]
        public static extern void TFE_OpSetAttrBool(SafeOpHandle op, string attr_name, bool value);

        /// <summary>
        /// 
        /// </summary>
        /// <param name="op">TFE_Op*</param>
        /// <param name="attr_name">const char*</param>
        /// <param name="value">const void*</param>
        /// <param name="length">size_t</param>
        [DllImport(TensorFlowLibName)]
        public static extern void TFE_OpSetAttrString(SafeOpHandle op, string attr_name, string value, uint length);
    
        [DllImport(TensorFlowLibName)]
        public static extern void TFE_OpSetAttrTypeList(SafeOpHandle op, string attr_name, TF_DataType[] values, int num_values);

        [DllImport(TensorFlowLibName)]
        public static extern void TFE_OpSetAttrValueProto(SafeOpHandle op, string attr_name, IMessage[] proto, int proto_len, SafeStatusHandle status);

        /// <summary>
        /// 
        /// </summary>
        /// <param name="op"></param>
        /// <param name="device_name"></param>
        /// <param name="status"></param>
        [DllImport(TensorFlowLibName)]
        public static extern void TFE_OpSetDevice(SafeOpHandle op, string device_name, SafeStatusHandle status);

        /// <summary>
        /// 
        /// </summary>
        /// <param name="op">TFE_Op*</param>
        /// <param name="h">TFE_TensorHandle*</param>
        /// <param name="status">TF_Status*</param>
        [DllImport(TensorFlowLibName)]
        public static extern void TFE_OpAddInput(SafeOpHandle op, SafeTensorHandleHandle h, SafeStatusHandle status);

        /// <summary>
        /// 
        /// </summary>
        /// <param name="t">const tensorflow::Tensor&amp;</param>
        /// <returns>TFE_TensorHandle*</returns>
        [DllImport(TensorFlowLibName)]
        public static extern SafeTensorHandleHandle TFE_NewTensorHandle(IntPtr t, SafeStatusHandle status);

        [DllImport(TensorFlowLibName)]
        public static extern SafeTensorHandleHandle TFE_EagerTensorHandle(IntPtr t);

        /// <summary>
        /// Sets the default execution mode (sync/async). Note that this can be
        /// overridden per thread using TFE_ContextSetExecutorForThread.
        /// </summary>
        /// <param name="opts">TFE_ContextOptions*</param>
        /// <param name="enable">unsigned char</param>
        [DllImport(TensorFlowLibName)]
        public static extern void TFE_ContextOptionsSetAsync(SafeContextOptionsHandle opts, byte enable);

        /// <summary>
        /// 
        /// </summary>
        /// <param name="h">TFE_TensorHandle*</param>
        /// <returns></returns>
        [DllImport(TensorFlowLibName)]
        public static extern TF_DataType TFE_TensorHandleDataType(SafeTensorHandleHandle h);

        /// <summary>
        /// This function will block till the operation that produces `h` has
        /// completed. The memory returned might alias the internal memory used by
        /// TensorFlow.
        /// </summary>
        /// <param name="h">TFE_TensorHandle*</param>
        /// <param name="status">TF_Status*</param>
        /// <returns></returns>
        [DllImport(TensorFlowLibName)]
        public static extern IntPtr TFE_TensorHandleResolve(SafeTensorHandleHandle h, SafeStatusHandle status);


        /// <summary>
        /// This function will block till the operation that produces `h` has completed.
        /// </summary>
        /// <param name="h">TFE_TensorHandle*</param>
        /// <param name="status">TF_Status*</param>
        /// <returns></returns>
        [DllImport(TensorFlowLibName)]
        public static extern int TFE_TensorHandleNumDims(SafeTensorHandleHandle h, SafeStatusHandle status);

        [DllImport(TensorFlowLibName)]
        public static extern int TFE_TensorHandleDim(SafeTensorHandleHandle h, int dim, SafeStatusHandle status);

        /// <summary>
        /// Returns the device of the operation that produced `h`. If `h` was produced by
        /// a copy, returns the destination device of the copy. Note that the returned
        /// device name is not always the device holding the tensor handle's memory. If
        /// you want the latter, use TFE_TensorHandleBackingDeviceName. This function
        /// will block till the operation that produces `h` has completed.
        /// </summary>
        /// <param name="h">TFE_TensorHandle*</param>
        /// <param name="status">TF_Status*</param>
        /// <returns></returns>
        [DllImport(TensorFlowLibName)]
        public static extern IntPtr TFE_TensorHandleDeviceName(SafeTensorHandleHandle h, SafeStatusHandle status);

        /// <summary>
        /// Returns the name of the device in whose memory `h` resides.
        /// </summary>
        /// <param name="h">TFE_TensorHandle*</param>
        /// <param name="status">TF_Status*</param>
        /// <returns></returns>
        [DllImport(TensorFlowLibName)]
        public static extern IntPtr TFE_TensorHandleBackingDeviceName(SafeTensorHandleHandle h, SafeStatusHandle status);

        /// <summary>
        /// 
        /// </summary>
        /// <param name="ctx">TFE_Context*</param>
        /// <param name="status">TF_Status*</param>
        /// <returns></returns>
        [DllImport(TensorFlowLibName)]
        public static extern SafeDeviceListHandle TFE_ContextListDevices(SafeContextHandle ctx, SafeStatusHandle status);

        /// <summary>
        /// 
        /// </summary>
        /// <param name="h">TFE_TensorHandle*</param>
        [DllImport(TensorFlowLibName)]
        public static extern void TFE_DeleteTensorHandle(IntPtr h);

        /// <summary>
        /// 
        /// </summary>
        /// <param name="h">TFE_TensorHandle*</param>
        [DllImport(TensorFlowLibName)]
        public static extern void TFE_DeleteEagerTensor(IntPtr h);

        [DllImport(TensorFlowLibName)]
        public static extern void TF_DeleteBindingArray(IntPtr h);

        [DllImport(TensorFlowLibName)]
        public static extern void TFE_DeleteBindingTensorArray(IntPtr h);

        /// <summary>
        /// Creates a new eager Executor. Nodes in one executor are guaranteed to be
        /// executed in sequence. Assigning nodes to different executors allows executing
        /// nodes in parallel.
        /// </summary>
        /// <param name="is_async"></param>
        /// <returns>TFE_Executor*</returns>
        [DllImport(TensorFlowLibName)]
        public static extern SafeExecutorHandle TFE_NewExecutor(bool is_async);

        /// <summary>
        /// Deletes the eager Executor without waiting for enqueued nodes. Please call
        /// TFE_ExecutorWaitForAllPendingNodes before calling this API if you want to
        /// make sure all nodes are finished.
        /// </summary>
        /// <param name="executor">TFE_Executor*</param>
        [DllImport(TensorFlowLibName)]
        public static extern void TFE_DeleteExecutor(IntPtr executor);

        /// <summary>
        /// Causes the calling thread to block till all ops dispatched in this executor
        /// have been executed. Note that "execution" here refers to kernel execution /
        /// scheduling of copies, etc. Similar to sync execution, it doesn't guarantee
        /// that lower level device queues (like GPU streams) have been flushed.
        /// 
        /// This call may not block for execution of ops enqueued concurrently with this
        /// call.
        /// </summary>
        /// <param name="executor">TFE_Executor*</param>
        /// <param name="status">TF_Status*</param>
        [DllImport(TensorFlowLibName)]
        public static extern void TFE_ExecutorWaitForAllPendingNodes(SafeExecutorHandle executor, SafeStatusHandle status);

        /// <summary>
        /// Sets a custom Executor for current thread. All nodes created by this thread
        /// will be added to this Executor. It will override current executor.
        /// </summary>
        /// <param name="ctx"></param>
        /// <param name="executor"></param>
        [DllImport(TensorFlowLibName)]
        public static extern void TFE_ContextSetExecutorForThread(SafeContextHandle ctx, SafeExecutorHandle executor);

        /// <summary>
        /// Returns the Executor for current thread.
        /// </summary>
        /// <param name="ctx"></param>
        /// <returns>TFE_Executor*</returns>
        [DllImport(TensorFlowLibName)]
        public static extern SafeExecutorHandle TFE_ContextGetExecutorForThread(SafeContextHandle ctx);

        [DllImport(TensorFlowLibName)]
        public static extern IntPtr TFE_TapeSetNew(bool persistent, bool watch_accessed_variables);

        [DllImport(TensorFlowLibName)]
        public static extern void TFE_TapeSetRemove(IntPtr tape);

        [DllImport(TensorFlowLibName)]
        public static extern void TFE_TapeWatch(IntPtr tape, IntPtr variable);

        [DllImport(TensorFlowLibName)]
        public static extern void TFE_TapeVariableAccessed(IntPtr variable);

        [DllImport(TensorFlowLibName)]
        public static extern IntPtr TFE_TapeWatchedVariables(IntPtr tape);

        [DllImport(TensorFlowLibName)]
        public static extern IntPtr ResourceVariable_Handle(IntPtr variable);

        [DllImport(TensorFlowLibName)]
        public static extern SafeStatusHandle TFE_TapeGradient(IntPtr tape, 
            IntPtr[] target, int target_size, 
            IntPtr[] sources, int source_size,
            IntPtr[] outputs, int output_size);
    }
}
