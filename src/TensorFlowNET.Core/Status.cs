using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow;

namespace TensorFlowNET.Core
{
    public class Status
    {
        private IntPtr _handle;
        public IntPtr Handle => _handle;

        public string ErrorMessage => c_api.TF_Message(_handle);

        public TF_Code Code => c_api.TF_GetCode(_handle);

        public Status()
        {
            _handle = c_api.TF_NewStatus();
        }
    }
}
