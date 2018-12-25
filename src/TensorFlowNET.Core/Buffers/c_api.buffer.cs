using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

namespace Tensorflow
{
    public static partial class c_api
    {
        [DllImport(TensorFlowLibName)]
        public static extern string TF_GetBuffer(IntPtr buffer);
    }
}
