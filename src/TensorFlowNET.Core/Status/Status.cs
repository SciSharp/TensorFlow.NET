using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow
{
    public class Status : IDisposable
    {
        private readonly IntPtr _handle;
        public IntPtr Handle => _handle;

        /// <summary>
        /// Error message
        /// </summary>
        public string Message => c_api.TF_Message(_handle);

        /// <summary>
        /// Error code
        /// </summary>
        public TF_Code Code => c_api.TF_GetCode(_handle);

        public Status()
        {
            _handle = c_api.TF_NewStatus();
        }

        public void SetStatus(TF_Code code, string msg)
        {
            c_api.TF_SetStatus(_handle, code, msg);
        }

        public void Dispose()
        {
            c_api.TF_DeleteStatus(_handle);
        }
    }
}
