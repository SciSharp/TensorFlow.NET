using System;
using System.Runtime.InteropServices;

namespace Tensorflow
{
    [StructLayout(LayoutKind.Sequential)]
    public struct TF_Input
    {
        public TF_Input(IntPtr oper, int index)
        {
            this.oper = oper;
            this.index = index;
        }

        public IntPtr oper;
        public int index;
    }
}
