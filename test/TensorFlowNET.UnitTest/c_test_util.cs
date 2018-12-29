using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;
using Tensorflow;
using Buffer = Tensorflow.Buffer;

namespace TensorFlowNET.UnitTest
{
    public static class c_test_util
    {
        public static bool GetAttrValue(Operation oper, string attr_name, ref AttrValue attr_value, Status s)
        {
            var buffer = new Buffer();
            c_api.TF_OperationGetAttrValueProto(oper, attr_name, buffer, s);
            attr_value = AttrValue.Parser.ParseFrom(buffer.Data);
            buffer.Dispose();
            return s.Code == TF_Code.TF_OK;
        }

        public static void PlaceholderHelper(Graph graph, Status s, string name, TF_DataType dtype, long[] dims, ref Operation op)
        {
            var desc = c_api.TF_NewOperation(graph, "Placeholder", name);
            c_api.TF_SetAttrType(desc, "dtype", dtype);
            if(dims != null)
            {
                c_api.TF_SetAttrShape(desc, "shape", dims, dims.Length);
            }
            op = c_api.TF_FinishOperation(desc, s);
            s.Check();
        }

        public static Operation Placeholder(Graph graph, Status s, string name = "feed", TF_DataType dtype = TF_DataType.TF_INT32, long[] dims = null)
        {
            Operation op = null;
            PlaceholderHelper(graph, s, name, dtype, dims, ref op);
            return op;
        }

        public static void ConstHelper(Tensor t, Graph graph, Status s, string name, ref Operation op)
        {
            var desc = c_api.TF_NewOperation(graph, "Const", name);
            c_api.TF_SetAttrTensor(desc, "value", t, s);
            s.Check();
            c_api.TF_SetAttrType(desc, "dtype", t.dtype);
            op = c_api.TF_FinishOperation(desc, s);
            s.Check();
            if(op == null)
            {
                throw new Exception("c_api.TF_FinishOperation failed.");
            }
        }

        public static Operation Const(Tensor t, Graph graph, Status s, string name)
        {
            Operation op = null;
            ConstHelper(t, graph, s, name, ref op);
            return op;
        }

        public static Operation ScalarConst(int v, Graph graph, Status s, string name = "Const")
        {
            return Const(new Tensor(v), graph, s, name);
        }
    }
}
