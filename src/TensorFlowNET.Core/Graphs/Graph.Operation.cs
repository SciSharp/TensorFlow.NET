using NumSharp.Core;
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

namespace Tensorflow
{
    public partial class Graph
    {
        public OpDef GetOpDef(string type)
        {
            using (var buffer = new Buffer())
            using (var status = new Status())
            {
                c_api.TF_GraphGetOpDef(_handle, type, buffer, status);
                return OpDef.Parser.ParseFrom(buffer.Data);
            }
        }

        public OperationDescription NewOperation(string opType, string opName)
        {
            OperationDescription desc = c_api.TF_NewOperation(_handle, opType, opName);
            return desc;

            /*c_api.TF_SetAttrTensor(desc, "value", tensor, Status);
            
            Status.Check();

            c_api.TF_SetAttrType(desc, "dtype", tensor.dtype);

            var op = c_api.TF_FinishOperation(desc, Status);
            Status.Check();

            return op;*/
        }
    }
}
