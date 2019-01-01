using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

namespace Tensorflow
{
    public class SessionOptions : IDisposable
    {
        private IntPtr _handle;

        public unsafe SessionOptions()
        {
            var opts = c_api.TF_NewSessionOptions();
            _handle = opts;
        }

        public unsafe SessionOptions(IntPtr handle)
        {
            _handle = handle;
        }

        public void Dispose()
        {
            c_api.TF_DeleteSessionOptions(_handle);
        }

        public static implicit operator IntPtr(SessionOptions opts) => opts._handle;
        public static implicit operator SessionOptions(IntPtr handle) => new SessionOptions(handle);
    }
}
