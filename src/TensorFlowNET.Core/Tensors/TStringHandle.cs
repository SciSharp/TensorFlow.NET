using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Util;

namespace Tensorflow
{
    public class TStringHandle : SafeTensorflowHandle
    {
        protected override bool ReleaseHandle()
        {
            c_api.TF_StringDealloc(handle);
            return true;
        }
    }
}
