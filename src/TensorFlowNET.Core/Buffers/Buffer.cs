using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

namespace Tensorflow
{
    public class Buffer
    {
        private IntPtr _handle;

        private TF_Buffer buffer;

        public byte[] Data;

        public int Length => (int)buffer.length;

        public unsafe Buffer(IntPtr handle)
        {
            _handle = handle;
            buffer = Marshal.PtrToStructure<TF_Buffer>(_handle);
            Data = new byte[buffer.length];
            if (buffer.length > 0)
                Marshal.Copy(buffer.data, Data, 0, (int)buffer.length);
        }
    }
}
