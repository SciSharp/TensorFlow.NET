using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;
using Tensorflow;

namespace TensorFlowNET.UnitTest
{
    public static class c_test_util
    {
        public static void ConstHelper(Tensor t, Graph graph, Status s, string name, ref IntPtr op)
        {
            var desc = c_api.TF_NewOperation(graph, "Const", name);
            c_api.TF_SetAttrTensor(desc, "value", t.Handle, s);
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
            IntPtr op = IntPtr.Zero;
            ConstHelper(t, graph, s, name, ref op);
            return new Operation(op);
        }

        public static Operation ScalarConst(int v, Graph graph, Status s, string name = "Const")
        {
            return Const(new Tensor(v), graph, s, name);
        }
    }
}
