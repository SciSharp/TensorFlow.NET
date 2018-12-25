using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

namespace Tensorflow
{
    public static partial class c_api
    {
        [DllImport(TensorFlowLibName)]
        public static extern void TF_GraphGetOpDef(IntPtr graph, string op_name, IntPtr output_op_def, IntPtr status);

        [DllImport(TensorFlowLibName)]
        public static unsafe extern IntPtr TF_NewGraph();
    }
}
