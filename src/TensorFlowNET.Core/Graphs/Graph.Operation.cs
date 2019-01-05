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
            return c_api.TF_NewOperation(_handle, opType, opName);
        }
    }
}
