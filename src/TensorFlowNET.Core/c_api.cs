using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

namespace Tensorflow
{
    /// <summary>
    /// C API for TensorFlow.
    /// 
    /// The API leans towards simplicity and uniformity instead of convenience
    /// since most usage will be by language specific wrappers.
    /// </summary>
    public static partial class c_api
    {
        public const string TensorFlowLibName = "tensorflow";

        public delegate void Deallocator(IntPtr data, IntPtr size, ref bool deallocatorData);

        [DllImport(TensorFlowLibName)]
        public static unsafe extern IntPtr TF_Version();
    }
}
