using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;
using size_t = System.IntPtr;

namespace Tensorflow
{
    [StructLayout(LayoutKind.Sequential)]
    public struct TF_Buffer
    {
        public IntPtr data;
        public size_t length;
        public IntPtr data_deallocator;
    }
}
