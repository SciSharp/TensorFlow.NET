using System;
using System.Runtime.InteropServices;

namespace Tensorflow
{
    [StructLayout(LayoutKind.Sequential)]
    public struct TF_OperationDescription
    {
        public IntPtr node_builder;
        public IntPtr graph;
        public IntPtr colocation_constraints;
    }
}
