using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Util;

namespace Tensorflow
{
    public sealed class SafeStringTensorHandle : SafeTensorHandle
    {
        Shape _shape;
        IntPtr _handle;
        const int TF_TSRING_SIZE = 24;

        protected SafeStringTensorHandle()
        {
        }

        public SafeStringTensorHandle(SafeTensorHandle handle, Shape shape)
            : base(handle.DangerousGetHandle())
        {
            _handle = c_api.TF_TensorData(handle);
            _shape = shape;
        }

        protected override bool ReleaseHandle()
        {
#if TRACK_TENSOR_LIFE
            print($"Delete StringTensorHandle 0x{handle.ToString("x16")}");
#endif

            for (int i = 0; i < _shape.size; i++)
            {
                c_api.TF_StringDealloc(_handle);
                _handle += TF_TSRING_SIZE;
            }

            SetHandle(IntPtr.Zero);

            return true;
        }
    }
}
