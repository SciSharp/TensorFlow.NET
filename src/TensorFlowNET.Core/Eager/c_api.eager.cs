using System;
using System.Runtime.InteropServices;

namespace Tensorflow
{
    public partial class c_api
    {
        /// <summary>
        /// Return a new options object.
        /// </summary>
        /// <returns>TFE_ContextOptions*</returns>
        [DllImport(TensorFlowLibName)]
        public static extern IntPtr TFE_NewContextOptions();

        /// <summary>
        /// Destroy an options object.
        /// </summary>
        /// <param name="options">TFE_ContextOptions*</param>
        [DllImport(TensorFlowLibName)]
        public static extern void TFE_DeleteContextOptions(IntPtr options);

        /// <summary>
        /// 
        /// </summary>
        /// <param name="opts">const TFE_ContextOptions*</param>
        /// <param name="status">TF_Status*</param>
        /// <returns>TFE_Context*</returns>
        [DllImport(TensorFlowLibName)]
        public static extern IntPtr TFE_NewContext(IntPtr opts, IntPtr status);

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
        public static extern void TFE_Execute(IntPtr op, IntPtr[] retvals, ref int num_retvals, IntPtr status);

        /// <summary>
        /// 
        /// </summary>
        /// <param name="ctx">TFE_Context*</param>
        /// <param name="op_or_function_name">const char*</param>
        /// <param name="status">TF_Status*</param>
        /// <returns></returns>
        [DllImport(TensorFlowLibName)]
        public static extern IntPtr TFE_NewOp(IntPtr ctx, string op_or_function_name, IntPtr status);

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
        public static extern void TFE_OpSetAttrShape(IntPtr op, string attr_name, long[] dims, int num_dims, Status out_status);

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
        public static extern void TFE_OpSetDevice(IntPtr op, string device_name, IntPtr status);

        /// <summary>
        /// 
        /// </summary>
        /// <param name="op">TFE_Op*</param>
        /// <param name="h">TFE_TensorHandle*</param>
        /// <param name="status">TF_Status*</param>
        [DllImport(TensorFlowLibName)]
        public static extern void TFE_OpAddInput(IntPtr op, IntPtr h, IntPtr status);

        /// <summary>
        /// 
        /// </summary>
        /// <param name="t">const tensorflow::Tensor&</param>
        /// <returns>TFE_TensorHandle*</returns>
        [DllImport(TensorFlowLibName)]
        public static extern IntPtr TFE_NewTensorHandle(IntPtr t, IntPtr status);

        /// <summary>
        /// Sets the default execution mode (sync/async). Note that this can be
        /// overridden per thread using TFE_ContextSetExecutorForThread.
        /// </summary>
        /// <param name="opts">TFE_ContextOptions*</param>
        /// <param name="enable">unsigned char</param>
        [DllImport(TensorFlowLibName)]
        public static extern void TFE_ContextOptionsSetAsync(IntPtr opts, byte enable);

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
        public static extern IntPtr TFE_TensorHandleResolve(IntPtr h, IntPtr status);

        /// <summary>
        /// This function will block till the operation that produces `h` has completed.
        /// </summary>
        /// <param name="h">TFE_TensorHandle*</param>
        /// <param name="status">TF_Status*</param>
        /// <returns></returns>
        [DllImport(TensorFlowLibName)]
        public static extern int TFE_TensorHandleNumDims(IntPtr h, IntPtr status);

        /// <summary>
        /// 
        /// </summary>
        /// <param name="ctx">TFE_Context*</param>
        /// <param name="status">TF_Status*</param>
        /// <returns></returns>
        [DllImport(TensorFlowLibName)]
        public static extern IntPtr TFE_ContextListDevices(IntPtr ctx, IntPtr status);

        /// <summary>
        /// 
        /// </summary>
        /// <param name="h">TFE_TensorHandle*</param>
        [DllImport(TensorFlowLibName)]
        public static extern void TFE_DeleteTensorHandle(IntPtr h);
    }
}
