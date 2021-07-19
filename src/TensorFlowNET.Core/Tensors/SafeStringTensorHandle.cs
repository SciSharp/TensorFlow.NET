using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Util;

namespace Tensorflow
{
    public sealed class SafeStringTensorHandle : SafeTensorHandle
    {
        Shape _shape;
        SafeTensorHandle _handle;
        const int TF_TSRING_SIZE = 24;

        protected SafeStringTensorHandle()
        {
        }

        public SafeStringTensorHandle(SafeTensorHandle handle, Shape shape)
            : base(handle.DangerousGetHandle())
        {
            _handle = handle;
            _shape = shape;
        }

        protected override bool ReleaseHandle()
        {
#if TRACK_TENSOR_LIFE
            print($"Delete StringTensorHandle 0x{handle.ToString("x16")}");
#endif

            long size = 1;
            foreach (var s in _shape.dims)
                size *= s;
            var tstr = c_api.TF_TensorData(_handle);

            for (int i = 0; i < size; i++)
            {
                c_api.TF_StringDealloc(tstr);
                tstr += TF_TSRING_SIZE;
            }

            SetHandle(IntPtr.Zero);

            return true;
        }
    }
}
