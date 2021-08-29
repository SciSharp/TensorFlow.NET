using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Util;

namespace Tensorflow.Lite
{
    public class SafeTfLiteModelHandle : SafeTensorflowHandle
    {
        protected SafeTfLiteModelHandle()
        {
        }

        public SafeTfLiteModelHandle(IntPtr handle)
            : base(handle)
        {
        }

        protected override bool ReleaseHandle()
        {
            c_api_lite.TfLiteModelDelete(handle);
            SetHandle(IntPtr.Zero);
            return true;
        }
    }
}
