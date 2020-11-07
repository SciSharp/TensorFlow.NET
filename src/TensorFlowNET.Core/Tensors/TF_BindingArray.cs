using System;
using System.Runtime.InteropServices;

namespace Tensorflow
{
    [StructLayout(LayoutKind.Sequential)]
    public struct TF_BindingArray
    {
        public IntPtr array;
        public int length;

        public static implicit operator TF_BindingArray(IntPtr handle)
            => handle == IntPtr.Zero ? default : Marshal.PtrToStructure<TF_BindingArray>(handle);

        public unsafe IntPtr this[int index]
            => array == IntPtr.Zero ? IntPtr.Zero : *((IntPtr*)array + index);

        public unsafe IntPtr[] Data
        {
            get
            {
                var results = new IntPtr[length];
                for (int i = 0; i < length; i++)
                    results[i] = array == IntPtr.Zero ? IntPtr.Zero : *((IntPtr*)array + i);
                return results;
            }
        }
    }
}
