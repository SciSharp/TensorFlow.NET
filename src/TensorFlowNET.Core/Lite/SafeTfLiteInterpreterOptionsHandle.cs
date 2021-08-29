using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Util;

namespace Tensorflow.Lite
{
    public class SafeTfLiteInterpreterOptionsHandle : SafeTensorflowHandle
    {
        protected SafeTfLiteInterpreterOptionsHandle()
        {
        }

        public SafeTfLiteInterpreterOptionsHandle(IntPtr handle)
            : base(handle)
        {
        }

        protected override bool ReleaseHandle()
        {
            c_api_lite.TfLiteInterpreterOptionsDelete(handle);
            SetHandle(IntPtr.Zero);
            return true;
        }
    }
}
