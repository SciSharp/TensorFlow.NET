using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

namespace Tensorflow
{
    public partial class c_api
    {
        [DllImport(TensorFlowLibName)]
        public static extern IntPtr TFE_NewResourceVariable();

        [DllImport(TensorFlowLibName)]
        public static extern void TFE_DeleteResourceVariable(IntPtr variable);

        [DllImport(TensorFlowLibName)]
        public static extern void TFE_SetResourceVariableHandle(IntPtr variable, IntPtr tensor);

        [DllImport(TensorFlowLibName)]
        public static extern void TFE_SetResourceVariableName(IntPtr variable, string name);
    }
}
