using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Util;

namespace Tensorflow.Lite
{
    public class SafeTfLiteInterpreterHandle : SafeTensorflowHandle
    {
        protected SafeTfLiteInterpreterHandle()
        {
        }

        public SafeTfLiteInterpreterHandle(IntPtr handle)
            : base(handle)
        {
        }

        protected override bool ReleaseHandle()
        {
            c_api_lite.TfLiteInterpreterDelete(handle);
            SetHandle(IntPtr.Zero);
            return true;
        }
    }
}
