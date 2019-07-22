using System;
using System.Runtime.InteropServices;

namespace Tensorflow
{
    [StructLayout(LayoutKind.Sequential)]
    public struct TF_ImportGraphDefResults
    {
        public IntPtr return_tensors;
        public IntPtr return_nodes;
        public IntPtr missing_unused_key_names;
        public IntPtr missing_unused_key_indexes;
        public IntPtr missing_unused_key_names_data;
    }
}
