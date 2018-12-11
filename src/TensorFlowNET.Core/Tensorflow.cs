using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

namespace TensorFlowNET.Core
{
    public static class Tensorflow
    {
        public const string TensorFlowLibName = "libtensorflow";

        [DllImport(TensorFlowLibName)]
        public static extern unsafe IntPtr TF_Version();

        public static string VERSION => Marshal.PtrToStringAnsi(TF_Version());
    }
}
