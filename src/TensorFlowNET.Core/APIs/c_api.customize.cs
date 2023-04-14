using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

namespace Tensorflow
{
    public partial class c_api
    {
        [DllImport(TensorFlowLibName)]
        public static extern void TFC_SetAttr(SafeGraphHandle graph, IntPtr op, string attr_name, SafeBufferHandle attr_value_proto, SafeStatusHandle status);
    }
}
