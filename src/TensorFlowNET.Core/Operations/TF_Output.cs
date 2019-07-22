using System;
using System.Runtime.InteropServices;

namespace Tensorflow
{
    [StructLayout(LayoutKind.Sequential)]
    public struct TF_Output
    {
        public TF_Output(IntPtr oper, int index)
        {
            this.oper = oper;
            this.index = index;
        }

        public IntPtr oper;
        public int index;
    }
}
