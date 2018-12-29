using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

namespace Tensorflow
{
    public static partial class c_api
    {
        [DllImport(TensorFlowLibName)]
        public static extern IntPtr TF_GetBuffer(TF_Buffer buffer);
    }
}
