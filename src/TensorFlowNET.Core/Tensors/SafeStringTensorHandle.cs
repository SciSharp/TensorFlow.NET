using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Util;

namespace Tensorflow
{
    public sealed class SafeStringTensorHandle : SafeTensorHandle
    {
        Shape _shape;
        SafeTensorHandle _tensorHandle;
        const int TF_TSRING_SIZE = 24;

        protected SafeStringTensorHandle()
        {
        }

        public SafeStringTensorHandle(SafeTensorHandle handle, Shape shape)
            : base(handle.DangerousGetHandle())
        {
            _tensorHandle = handle;
            _shape = shape;
            bool success = false;
            _tensorHandle.DangerousAddRef(ref success);
        }

        protected override bool ReleaseHandle()
        {
            var _handle = c_api.TF_TensorData(_tensorHandle);
#if TRACK_TENSOR_LIFE
            Console.WriteLine($"Delete StringTensorData 0x{_handle.ToString("x16")}");
#endif
            for (int i = 0; i < _shape.size; i++)
            {
                c_api.TF_StringDealloc(_handle);
                _handle += TF_TSRING_SIZE;
            }

            SetHandle(IntPtr.Zero);
            _tensorHandle.DangerousRelease();

            return true;
        }
    }
}
