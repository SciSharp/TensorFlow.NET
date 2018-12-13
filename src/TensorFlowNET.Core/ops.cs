using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;
using Tensorflow;
using tf = TensorFlowNET.Core.Tensorflow;
using TF_DataType = Tensorflow.DataType;
using node_def_pb2 = Tensorflow;

namespace TensorFlowNET.Core
{
    public static class ops
    {
        public static Graph get_default_graph()
        {
            return tf.Graph();
        }

        public static unsafe IntPtr _create_c_op(Graph graph, NodeDef node_def, object inputs)
        {
            var op_desc = c_api.TF_NewOperation(graph.handle, node_def.Op, node_def.Name);
            var status = c_api.TF_NewStatus();

            // Doesn't work
            /*foreach(var attr in node_def.Attr)
            {
                if (attr.Value.Tensor != null)
                {
                    switch (attr.Value.Tensor.Dtype)
                    {
                        case DataType.DtDouble:
                            var proto = (double*)Marshal.AllocHGlobal(sizeof(double));
                            *proto = attr.Value.Tensor.DoubleVal[0];
                            c_api.TF_SetAttrValueProto(op_desc, attr.Key, proto, proto_len: (UIntPtr)sizeof(double), status: status);
                            break;
                    }
                }
                else
                {
                    //c_api.TF_SetAttrValueProto(op_desc, attr.Key, null, proto_len: UIntPtr.Zero, status: status);
                }
            } */

            foreach (var attr in node_def.Attr)
            {
                if (attr.Value.Tensor == null) continue;
                switch (attr.Value.Tensor.Dtype)
                {
                    case DataType.DtDouble:
                        var v = (double*)Marshal.AllocHGlobal(sizeof(double));
                        *v = attr.Value.Tensor.DoubleVal[0];
                        var tensor = c_api.TF_NewTensor(TF_DataType.DtDouble, 0, 0, data: (IntPtr)v, len: (UIntPtr)sizeof(double), deallocator: Tensorflow.FreeTensorDataDelegate, deallocator_arg: IntPtr.Zero);
                        c_api.TF_SetAttrTensor(op_desc, "value", tensor, status);
                        c_api.TF_SetAttrType(op_desc, "dtype", TF_DataType.DtDouble);
                        break;
                    case DataType.DtString:
                        
                        var proto = Marshal.StringToHGlobalAnsi(attr.Value.Tensor.StringVal[0].ToStringUtf8());
                        c_api.TF_SetAttrValueProto(op_desc, attr.Key, proto.ToPointer(), proto_len: (UIntPtr)32, status: status);
                        break;
                }
            }

            var c_op = c_api.TF_FinishOperation(op_desc, status);

            return c_op;
        }

        public static NodeDef _NodeDef(string op_type, string name, string device = "", Dictionary<string, AttrValue> attrs = null)
        {
            var node_def = new node_def_pb2.NodeDef();
            node_def.Op = op_type;
            node_def.Name = name;

            foreach (var attr in attrs)
            {
                node_def.Attr.Add(attr.Key, attr.Value);
            }
            
            return node_def;
        }

        public static int uid()
        {
            return 1;
        }
    }
}
