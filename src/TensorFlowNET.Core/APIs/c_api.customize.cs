using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

namespace Tensorflow
{
    public partial class c_api
    {
        [DllImport(TensorFlowLibName)]
        public static extern void TF_SetAttr(SafeGraphHandle graph, IntPtr op, string attr_name, SafeBufferHandle attr_value_proto, SafeStatusHandle status);
        [DllImport(TensorFlowLibName)]
        public static extern SafeBufferHandle TF_GetHandleShapeAndType(SafeGraphHandle c_graph, TF_Output output);
        [DllImport(TensorFlowLibName)]
        public static extern void TF_SetHandleShapeAndType(SafeGraphHandle c_graph, TF_Output output, byte[] data, long proto_len, SafeStatusHandle status);
    }
}
