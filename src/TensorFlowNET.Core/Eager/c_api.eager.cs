﻿using System;
using System.Runtime.InteropServices;
using Tensorflow.Device;
using Tensorflow.Eager;
using TFE_Executor = System.IntPtr;

namespace Tensorflow
{
    public partial class c_api
    {
        [DllImport(TensorFlowLibName)]
        public static extern void TFE_RegisterGradientFunction(gradient_function_callback gradientFunctionCallback,
            delete_backward_function_callback deleteBackwardFunctionCallback);

        /// <summary>
        /// 
        /// </summary>
        /// <param name="op_name"></param>
        /// <param name="op_inputs"></param>
        /// <param name="op_outputs"></param>
        /// <param name="num_attrs"></param>
        /// <param name="output_grads">previous node ouput</param>
        /// <param name="skip_input_indices"></param>
        /// <returns></returns>
        [UnmanagedFunctionPointer(CallingConvention.StdCall)]
        public delegate IntPtr gradient_function_callback(string op_name,
            IntPtr op_inputs,
            IntPtr op_outputs,
            string attrs_string,
            IntPtr output_grads,
            IntPtr skip_input_indices);

        [UnmanagedFunctionPointer(CallingConvention.StdCall)]
        public delegate void delete_backward_function_callback(string op_name,
            IntPtr op_inputs,
            IntPtr op_outputs);

        [DllImport(TensorFlowLibName)]
        public static extern IntPtr TFE_WrapGradientResult(IntPtr[] gradients, int num_gradients);

        [DllImport(TensorFlowLibName)]
        public static extern IntPtr VSpace_Handle(VSpace_callback_Ones ones, VSpace_callback_AggregateGrads aggregate_grads);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)]
        public delegate IntPtr VSpace_callback_Ones(long[] shape, int dims, TF_DataType dtype);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)]
        public delegate IntPtr VSpace_callback_AggregateGrads(TF_BindingArray gradients);

        [DllImport(TensorFlowLibName)]
        public static extern void TFE_RegisterVSpace(IntPtr vspace);
        
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
        public static extern TF_AttrType TFE_OpGetAttrType(IntPtr op, string attr_name, ref byte is_list, SafeStatusHandle status);

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
        public static extern int TFE_OpGetInputLength(IntPtr op, string input_name, SafeStatusHandle status);

        /// <summary>
        /// Returns the length (number of tensors) of the output argument `output_name`
        /// found in the provided `op`.
        /// </summary>
        /// <param name="op"></param>
        /// <param name="input_name"></param>
        /// <param name="status"></param>
        /// <returns></returns>
        [DllImport(TensorFlowLibName)]
        public static extern int TFE_OpGetOutputLength(IntPtr op, string input_name, SafeStatusHandle status);

        /// <summary>
        /// 
        /// </summary>
        /// <param name="op">TFE_Op*</param>
        /// <param name="inputs">TFE_TensorHandle**</param>
        /// <param name="num_inputs">int</param>
        /// <param name="status">TF_Status*</param>
        /// <returns></returns>
        [DllImport(TensorFlowLibName)]
        public static extern int TFE_OpAddInputList(IntPtr op, IntPtr[] inputs, int num_inputs, SafeStatusHandle status);

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

        /// <summary>
        /// Execute the operation defined by 'op' and return handles to computed
        /// tensors in `retvals`.
        /// </summary>
        /// <param name="op">TFE_Op*</param>
        /// <param name="retvals">TFE_TensorHandle**</param>
        /// <param name="num_retvals">int*</param>
        /// <param name="status">TF_Status*</param>
        [DllImport(TensorFlowLibName)]
        public static extern void TFE_Execute(IntPtr op, IntPtr[] retvals, ref int num_retvals, SafeStatusHandle status);

        /// <summary>
        /// 
        /// </summary>
        /// <param name="ctx">TFE_Context*</param>
        /// <param name="op_or_function_name">const char*</param>
        /// <param name="status">TF_Status*</param>
        /// <returns></returns>
        [DllImport(TensorFlowLibName)]
        public static extern TFE_Op TFE_NewOp(SafeContextHandle ctx, string op_or_function_name, SafeStatusHandle status);

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
        public static extern void TFE_OpReset(IntPtr op_to_reset, string op_or_function_name, string raw_device_name, SafeStatusHandle status);

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
        public static extern void TFE_OpSetAttrType(IntPtr op, string attr_name, TF_DataType value);

        [DllImport(TensorFlowLibName)]
        public static extern void TFE_OpSetAttrInt(IntPtr op, string attr_name, long value);

        /// <summary>
        /// 
        /// </summary>
        /// <param name="op">TFE_Op*</param>
        /// <param name="attr_name">const char*</param>
        /// <param name="dims">const int64_t*</param>
        /// <param name="num_dims">const int</param>
        /// <param name="out_status">TF_Status*</param>
        [DllImport(TensorFlowLibName)]
        public static extern void TFE_OpSetAttrShape(IntPtr op, string attr_name, long[] dims, int num_dims, SafeStatusHandle out_status);

        [DllImport(TensorFlowLibName)]
        public static extern void TFE_OpSetAttrBool(IntPtr op, string attr_name, bool value);

        /// <summary>
        /// 
        /// </summary>
        /// <param name="op">TFE_Op*</param>
        /// <param name="attr_name">const char*</param>
        /// <param name="value">const void*</param>
        /// <param name="length">size_t</param>
        [DllImport(TensorFlowLibName)]
        public static extern void TFE_OpSetAttrString(IntPtr op, string attr_name, string value, uint length);

        /// <summary>
        /// 
        /// </summary>
        /// <param name="op"></param>
        /// <param name="device_name"></param>
        /// <param name="status"></param>
        [DllImport(TensorFlowLibName)]
        public static extern void TFE_OpSetDevice(TFE_Op op, string device_name, SafeStatusHandle status);

        /// <summary>
        /// 
        /// </summary>
        /// <param name="op">TFE_Op*</param>
        /// <param name="h">TFE_TensorHandle*</param>
        /// <param name="status">TF_Status*</param>
        [DllImport(TensorFlowLibName)]
        public static extern void TFE_OpAddInput(IntPtr op, IntPtr h, SafeStatusHandle status);

        /// <summary>
        /// 
        /// </summary>
        /// <param name="t">const tensorflow::Tensor&</param>
        /// <returns>TFE_TensorHandle*</returns>
        [DllImport(TensorFlowLibName)]
        public static extern TFE_TensorHandle TFE_NewTensorHandle(IntPtr t, SafeStatusHandle status);

        [DllImport(TensorFlowLibName)]
        public static extern IntPtr TFE_EagerTensorHandle(IntPtr t);

        [DllImport(TensorFlowLibName)]
        public static extern int TFE_EagerTensorId(IntPtr t);

        [DllImport(TensorFlowLibName)]
        public static extern IntPtr TFE_NewEagerTensor();

        [DllImport(TensorFlowLibName)]
        public static extern void TFE_SetEagerTensorHandle(IntPtr tensor, IntPtr handle);

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
        public static extern TF_DataType TFE_TensorHandleDataType(IntPtr h);

        /// <summary>
        /// This function will block till the operation that produces `h` has
        /// completed. The memory returned might alias the internal memory used by
        /// TensorFlow.
        /// </summary>
        /// <param name="h">TFE_TensorHandle*</param>
        /// <param name="status">TF_Status*</param>
        /// <returns></returns>
        [DllImport(TensorFlowLibName)]
        public static extern IntPtr TFE_TensorHandleResolve(IntPtr h, SafeStatusHandle status);


        /// <summary>
        /// This function will block till the operation that produces `h` has completed.
        /// </summary>
        /// <param name="h">TFE_TensorHandle*</param>
        /// <param name="status">TF_Status*</param>
        /// <returns></returns>
        [DllImport(TensorFlowLibName)]
        public static extern int TFE_TensorHandleNumDims(IntPtr h, SafeStatusHandle status);

        [DllImport(TensorFlowLibName)]
        public static extern int TFE_TensorHandleDim(IntPtr h, int dim, SafeStatusHandle status);

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
        public static extern IntPtr TFE_TensorHandleDeviceName(IntPtr h, SafeStatusHandle status);

        /// <summary>
        /// Returns the name of the device in whose memory `h` resides.
        /// </summary>
        /// <param name="h">TFE_TensorHandle*</param>
        /// <param name="status">TF_Status*</param>
        /// <returns></returns>
        [DllImport(TensorFlowLibName)]
        public static extern IntPtr TFE_TensorHandleBackingDeviceName(IntPtr h, SafeStatusHandle status);

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
        public static extern TFE_Executor TFE_NewExecutor(bool is_async);

        /// <summary>
        /// Deletes the eager Executor without waiting for enqueued nodes. Please call
        /// TFE_ExecutorWaitForAllPendingNodes before calling this API if you want to
        /// make sure all nodes are finished.
        /// </summary>
        /// <param name="e">TFE_Executor*</param>
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
        public static extern void TFE_ExecutorWaitForAllPendingNodes(TFE_Executor executor, SafeStatusHandle status);

        /// <summary>
        /// Sets a custom Executor for current thread. All nodes created by this thread
        /// will be added to this Executor. It will override current executor.
        /// </summary>
        /// <param name="ctx"></param>
        /// <param name="executor"></param>
        [DllImport(TensorFlowLibName)]
        public static extern void TFE_ContextSetExecutorForThread(SafeContextHandle ctx, TFE_Executor executor);

        /// <summary>
        /// Returns the Executor for current thread.
        /// </summary>
        /// <param name="ctx"></param>
        /// <returns>TFE_Executor*</returns>
        [DllImport(TensorFlowLibName)]
        public static extern TFE_Executor TFE_ContextGetExecutorForThread(SafeContextHandle ctx);

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
