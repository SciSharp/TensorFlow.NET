using System;
using System.Runtime.InteropServices;

namespace Tensorflow
{
    [StructLayout(LayoutKind.Sequential)]
    public struct TF_Tensor
    {
        public TF_DataType dtype;
        public IntPtr shape;
        public IntPtr buffer;
    }
}
