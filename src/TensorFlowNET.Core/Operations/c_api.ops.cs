using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

namespace Tensorflow
{
    public static partial class c_api
    {
        /// <summary>
        /// For inputs that take a single tensor.
        /// </summary>
        /// <param name="desc"></param>
        /// <param name="input"></param>
        [DllImport(TensorFlowLibName)]
        public static unsafe extern void TF_AddInput(TF_OperationDescription desc, TF_Output input);

        [DllImport(TensorFlowLibName)]
        public static unsafe extern IntPtr TF_FinishOperation(TF_OperationDescription desc, IntPtr status);

        [DllImport(TensorFlowLibName)]
        public static unsafe extern TF_OperationDescription TF_NewOperation(IntPtr graph, string opType, string oper_name);

        [DllImport(TensorFlowLibName)]
        public static extern unsafe int TF_OperationNumOutputs(IntPtr oper);

        [DllImport(TensorFlowLibName)]
        public static extern unsafe void TF_SetAttrValueProto(TF_OperationDescription desc, string attr_name, IntPtr proto, UIntPtr proto_len, IntPtr status);

        [DllImport(TensorFlowLibName)]
        public static extern unsafe void TF_SetAttrTensor(TF_OperationDescription desc, string attr_name, IntPtr value, IntPtr status);

        [DllImport(TensorFlowLibName)]
        public static extern unsafe void TF_SetAttrType(TF_OperationDescription desc, string attr_name, TF_DataType value);
    }
}
