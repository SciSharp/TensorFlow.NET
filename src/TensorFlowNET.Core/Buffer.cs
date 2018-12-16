using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;
using Tensorflow;

namespace TensorFlowNET.Core
{
    public class Buffer
    {
        private IntPtr _handle;
        public IntPtr Handle => _handle;
        //public TF_Buffer buffer => Marshal.PtrToStructure<TF_Buffer>(_handle);

        public unsafe Buffer()
        {
            _handle = Marshal.AllocHGlobal(sizeof(TF_Buffer));
        }

        public byte[] GetBuffer()
        {
            var buffer = Marshal.PtrToStructure<TF_Buffer>(_handle);

            var data = Marshal.AllocHGlobal(buffer.length);
            //var bytes = c_api.TF_GetBuffer(buffer.data);

            return null;
        }
    }
}
