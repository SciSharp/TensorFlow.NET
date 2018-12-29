using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

namespace Tensorflow
{
    public static partial class c_api
    {
        /// <summary>
        /// Get the OpList of all OpDefs defined in this address space.
        /// </summary>
        /// <returns></returns>
        [DllImport(TensorFlowLibName)]
        public static unsafe extern IntPtr TF_GetAllOpList();

        /// <summary>
        /// For inputs that take a single tensor.
        /// </summary>
        /// <param name="desc"></param>
        /// <param name="input"></param>
        [DllImport(TensorFlowLibName)]
        public static unsafe extern void TF_AddInput(IntPtr desc, TF_Output input);

        [DllImport(TensorFlowLibName)]
        public static unsafe extern IntPtr TF_FinishOperation(IntPtr desc, IntPtr status);

        [DllImport(TensorFlowLibName)]
        public static unsafe extern IntPtr TF_NewOperation(IntPtr graph, string opType, string oper_name);

        [DllImport(TensorFlowLibName)]
        public static extern unsafe int TF_OperationNumOutputs(IntPtr oper);

        [DllImport(TensorFlowLibName)]
        public static extern unsafe void TF_SetAttrValueProto(IntPtr desc, string attr_name, IntPtr proto, UIntPtr proto_len, IntPtr status);

        [DllImport(TensorFlowLibName)]
        public static extern unsafe void TF_SetAttrTensor(IntPtr desc, string attr_name, IntPtr value, IntPtr status);

        [DllImport(TensorFlowLibName)]
        public static extern unsafe void TF_SetAttrType(IntPtr desc, string attr_name, TF_DataType value);
    }
}
