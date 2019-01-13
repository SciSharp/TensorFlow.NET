using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

namespace Tensorflow
{
    public partial class c_api
    {
        [DllImport(TensorFlowLibName)]
        public static extern void TF_DeleteBuffer(IntPtr buffer);

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
