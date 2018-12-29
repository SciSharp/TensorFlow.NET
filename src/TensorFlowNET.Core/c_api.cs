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
    /// 
    /// The params type mapping between .net and c_api
    /// TF_XX** => ref IntPtr (TF_Operation** op) => (ref IntPtr op)
    /// TF_XX* => IntPtr (TF_Graph* graph) => (IntPtr graph)
    /// struct => struct (TF_Output output) => (TF_Output output)
    /// const char* => string
    /// int32_t => int
    /// int64_t* => long[]
    /// size_t* => unlong[]
    /// void* => IntPtr
    /// </summary>
    public static partial class c_api
    {
        public const string TensorFlowLibName = "tensorflow";

        public delegate void Deallocator(IntPtr data, IntPtr size, ref bool deallocator);

        [DllImport(TensorFlowLibName)]
        public static unsafe extern IntPtr TF_Version();
    }
}
