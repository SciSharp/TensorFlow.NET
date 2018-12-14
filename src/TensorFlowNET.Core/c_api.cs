using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

using size_t = System.UIntPtr;
using TF_Graph = System.IntPtr;
using TF_OperationDescription = System.IntPtr;
using TF_Operation = System.IntPtr;
using TF_Status = System.IntPtr;
using TF_Tensor = System.IntPtr;

using TF_DataType = Tensorflow.DataType;
using Tensorflow;
using static TensorFlowNET.Core.Tensorflow;

namespace TensorFlowNET.Core
{
    public static class c_api
    {
        public const string TensorFlowLibName = "tensorflow";

        [DllImport(TensorFlowLibName)]
        public static unsafe extern TF_Operation TF_FinishOperation(TF_OperationDescription desc, TF_Status status);

        [DllImport(TensorFlowLibName)]
        public static extern unsafe TF_Code TF_GetCode(TF_Status s);

        [DllImport(TensorFlowLibName)]
        public static extern unsafe string TF_Message(TF_Status s);

        [DllImport(TensorFlowLibName)]
        public static unsafe extern IntPtr TF_NewGraph();

        [DllImport(TensorFlowLibName)]
        public static unsafe extern TF_OperationDescription TF_NewOperation(TF_Graph graph, string opType, string oper_name);

        [DllImport(TensorFlowLibName)]
        public static unsafe extern TF_Status TF_NewStatus();

        [DllImport(TensorFlowLibName)]
        public static extern unsafe TF_Tensor TF_NewTensor(TF_DataType dataType, Int64 dims, int num_dims, IntPtr data, size_t len, Deallocator deallocator, IntPtr deallocator_arg);

        [DllImport(TensorFlowLibName)]
        public static extern unsafe int TF_OperationNumOutputs(TF_Operation oper);

        [DllImport(TensorFlowLibName)]
        public static extern unsafe void TF_SetAttrValueProto(TF_OperationDescription desc, string attr_name, IntPtr proto, size_t proto_len, TF_Status status);

        [DllImport(TensorFlowLibName)]
        public static extern unsafe void TF_SetAttrTensor(TF_OperationDescription desc, string attr_name, TF_Tensor value, TF_Status status);

        [DllImport(TensorFlowLibName)]
        public static extern unsafe void TF_SetAttrType(TF_OperationDescription desc, string attr_name, TF_DataType value);

        [DllImport(TensorFlowLibName)]
        public static unsafe extern IntPtr TF_Version();
    }
}
