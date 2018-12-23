using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

using size_t = System.UIntPtr;
using TF_Graph = System.IntPtr;
using TF_Operation = System.IntPtr;
using TF_Status = System.IntPtr;
using TF_Session = System.IntPtr;
using TF_SessionOptions = System.IntPtr;

namespace Tensorflow
{
    public static class c_api
    {
        public const string TensorFlowLibName = "tensorflow";

        /// <summary>
        /// For inputs that take a single tensor.
        /// </summary>
        /// <param name="desc"></param>
        /// <param name="input"></param>
        [DllImport(TensorFlowLibName)]
        public static unsafe extern void TF_AddInput(TF_OperationDescription desc, TF_Output input);

        [DllImport(TensorFlowLibName)]
        public static unsafe extern void TF_DeleteSessionOptions(TF_SessionOptions opts);

        [DllImport(TensorFlowLibName)]
        public static unsafe extern TF_Operation TF_FinishOperation(TF_OperationDescription desc, TF_Status status);

        [DllImport(TensorFlowLibName)]
        public static extern string TF_GetBuffer(IntPtr buffer);

        [DllImport(TensorFlowLibName)]
        public static extern unsafe TF_Code TF_GetCode(TF_Status s);

        [DllImport(TensorFlowLibName)]
        public static extern void TF_GraphGetOpDef(TF_Graph graph, string op_name, IntPtr output_op_def, TF_Status status);

        [DllImport(TensorFlowLibName)]
        public static extern unsafe string TF_Message(TF_Status s);

        [DllImport(TensorFlowLibName)]
        public static unsafe extern TF_Graph TF_NewGraph();

        [DllImport(TensorFlowLibName)]
        public static unsafe extern TF_OperationDescription TF_NewOperation(TF_Graph graph, string opType, string oper_name);

        [DllImport(TensorFlowLibName)]
        public static unsafe extern TF_Status TF_NewStatus();

        /// <summary>
        /// Return a new tensor that holds the bytes data[0,len-1]
        /// </summary>
        /// <param name="dataType"></param>
        /// <param name="dims"></param>
        /// <param name="num_dims"></param>
        /// <param name="data"></param>
        /// <param name="len">num_bytes, ex: 6 * sizeof(float)</param>
        /// <param name="deallocator"></param>
        /// <param name="deallocator_arg"></param>
        /// <returns></returns>
        [DllImport(TensorFlowLibName)]
        public static extern unsafe IntPtr TF_NewTensor(TF_DataType dataType, long[] dims, int num_dims, IntPtr data, size_t len, tf.Deallocator deallocator, IntPtr deallocator_arg);

        [DllImport(TensorFlowLibName)]
        public static extern unsafe int TF_OperationNumOutputs(TF_Operation oper);

        [DllImport(TensorFlowLibName)]
        public static extern unsafe void TF_SetAttrValueProto(TF_OperationDescription desc, string attr_name, IntPtr proto, size_t proto_len, TF_Status status);

        [DllImport(TensorFlowLibName)]
        public static extern unsafe void TF_SetAttrTensor(TF_OperationDescription desc, string attr_name, TF_Tensor value, TF_Status status);

        /// <summary>
        /// 
        /// </summary>
        /// <param name="session"></param>
        /// <param name="run_options"></param>
        /// <param name="inputs"></param>
        /// <param name="input_values"></param>
        /// <param name="ninputs"></param>
        /// <param name="outputs"></param>
        /// <param name="output_values"></param>
        /// <param name="noutputs"></param>
        /// <param name="target_opers"></param>
        /// <param name="ntargets"></param>
        /// <param name="run_metadata"></param>
        /// <param name="status"></param>
        [DllImport(TensorFlowLibName)]
        public static extern unsafe void TF_SessionRun(TF_Session session, IntPtr run_options,
                   TF_Output[] inputs, IntPtr[] input_values, int ninputs, 
                   TF_Output[] outputs, IntPtr[] output_values, int noutputs,
                   TF_Operation[] target_opers, int ntargets,
                   IntPtr run_metadata, 
                   TF_Status status);

        [DllImport(TensorFlowLibName)]
        public static extern unsafe void TF_SetAttrType(TF_OperationDescription desc, string attr_name, TF_DataType value);

        [DllImport(TensorFlowLibName)]
        public static extern unsafe IntPtr TF_TensorData(IntPtr tensor);

        [DllImport(TensorFlowLibName)]
        public static extern unsafe TF_DataType TF_TensorType(IntPtr tensor);

        [DllImport(TensorFlowLibName)]
        public static extern TF_Session TF_NewSession(TF_Graph graph, TF_SessionOptions opts, TF_Status status);

        [DllImport(TensorFlowLibName)]
        public static extern TF_SessionOptions TF_NewSessionOptions();

        [DllImport(TensorFlowLibName)]
        public static unsafe extern IntPtr TF_Version();
    }
}
