using System;
using System.Runtime.InteropServices;
using Tensorflow.Variables;

namespace Tensorflow
{
    public partial class c_api
    {
        [DllImport(TensorFlowLibName)]
        public static extern SafeResourceVariableHandle TFE_NewResourceVariable();

        [DllImport(TensorFlowLibName)]
        public static extern void TFE_DeleteResourceVariable(IntPtr variable);

        [DllImport(TensorFlowLibName)]
        public static extern void TFE_SetResourceVariableHandle(SafeResourceVariableHandle variable, IntPtr tensor);

        [DllImport(TensorFlowLibName)]
        public static extern void TFE_SetResourceVariableName(SafeResourceVariableHandle variable, string name);
    }
}
