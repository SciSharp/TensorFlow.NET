using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;
using tf = TensorFlowNET.Core.Tensorflow;
using TF_DataType = Tensorflow.DataType;

namespace TensorFlowNET.Core
{
    public static class ops
    {
        public static Graph get_default_graph()
        {
            return tf.Graph();
        }

        public static unsafe IntPtr _create_c_op(Graph graph, object inputs)
        {
            var op_desc = c_api.TF_NewOperation(graph.handle, "Const", "Const0");
            var status = c_api.TF_NewStatus();

            IntPtr tensor = IntPtr.Zero;

            switch (inputs)
            {
                case double value:
                    var v = (double*)Marshal.AllocHGlobal(sizeof(double));
                    *v = value;
                    tensor = c_api.TF_NewTensor(TF_DataType.DtDouble, 0, 0, data: (IntPtr)v, len: (UIntPtr)sizeof(double), deallocator: Tensorflow.FreeTensorDataDelegate, deallocator_arg: IntPtr.Zero);
                    c_api.TF_SetAttrType(op_desc, "dtype", TF_DataType.DtDouble);
                    break;
            }

            c_api.TF_SetAttrTensor(op_desc, "value", tensor, status);

            var c_op = c_api.TF_FinishOperation(op_desc, status);

            return c_op;
        }
    }
}
