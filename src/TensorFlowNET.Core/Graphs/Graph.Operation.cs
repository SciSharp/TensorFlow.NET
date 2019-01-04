using NumSharp.Core;
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

namespace Tensorflow
{
    public partial class Graph
    {
        public Operation NewOperation(string opType, string opName, Tensor tensor)
        {
            var desc = c_api.TF_NewOperation(_handle, opType, opName);

            if (tensor.dtype == TF_DataType.TF_STRING)
            {
                var value = "Hello World!";
                var bytes = Encoding.UTF8.GetBytes(value);
                var buf = Marshal.AllocHGlobal(bytes.Length + 1);
                Marshal.Copy(bytes, 0, buf, bytes.Length);
                c_api.TF_SetAttrString(desc, "value", buf, (uint)value.Length);
            }
            else
            {
                c_api.TF_SetAttrTensor(desc, "value", tensor, Status);
            }
            
            Status.Check();

            c_api.TF_SetAttrType(desc, "dtype", tensor.dtype);

            var op = c_api.TF_FinishOperation(desc, Status);
            Status.Check();

            return op;
        }
    }
}
