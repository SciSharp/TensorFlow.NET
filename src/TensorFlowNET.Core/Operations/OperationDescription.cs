using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow
{
    public class OperationDescription
    {
        private IntPtr _handle;

        public OperationDescription(IntPtr handle)
        {
            _handle = handle;
        }

        public void AddInputList(params TF_Output[] inputs)
        {
            c_api.TF_AddInputList(_handle, inputs, inputs.Length);
        }

        public static implicit operator OperationDescription(IntPtr handle)
        {
            return new OperationDescription(handle);
        }

        public static implicit operator IntPtr(OperationDescription desc)
        {
            return desc._handle;
        }
    }
}
