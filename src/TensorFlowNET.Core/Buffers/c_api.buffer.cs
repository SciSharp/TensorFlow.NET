using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

namespace Tensorflow
{
    public static partial class c_api
    {
        /// <summary>
        /// Useful for passing *out* a protobuf.
        /// </summary>
        /// <returns></returns>
        [DllImport(TensorFlowLibName)]
        public static extern IntPtr TF_NewBuffer();

        [DllImport(TensorFlowLibName)]
        public static extern IntPtr TF_GetBuffer(TF_Buffer buffer);
    }
}
