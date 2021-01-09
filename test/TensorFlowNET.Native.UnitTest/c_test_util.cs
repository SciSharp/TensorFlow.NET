using System;
using System.Diagnostics.CodeAnalysis;
using Tensorflow.Util;

namespace Tensorflow.Native.UnitTest
{
    /// <summary>
    /// Port from `tensorflow\c\c_test_util.cc`
    /// </summary>
    public static class c_test_util
    {
        public static Operation Add(Operation l, Operation r, Graph graph, Status s, string name = "add")
        {
            lock (Locks.ProcessWide)
            {
                var desc = c_api.TF_NewOperation(graph, "AddN", name);

                var inputs = new TF_Output[]
                {
                    new TF_Output(l, 0),
                    new TF_Output(r, 0),
                };

                c_api.TF_AddInputList(desc, inputs, inputs.Length);

                var op = c_api.TF_FinishOperation(desc, s.Handle);
                s.Check();

                return op;
            }
        }

        [SuppressMessage("ReSharper", "RedundantAssignment")]
        public static bool GetAttrValue(Operation oper, string attr_name, ref AttrValue attr_value, Status s)
        {
            lock (Locks.ProcessWide)
            {
                using (var buffer = new Buffer())
                {
                    c_api.TF_OperationGetAttrValueProto(oper, attr_name, buffer.Handle, s.Handle);
                    attr_value = AttrValue.Parser.ParseFrom(buffer.DangerousMemoryBlock.Stream());
                }

                return s.Code == TF_Code.TF_OK;
            }
        }

        public static GraphDef GetGraphDef(Graph graph)
        {
            lock (Locks.ProcessWide)
            {
                using (var s = new Status())
                using (var buffer = new Buffer())
                {
                    c_api.TF_GraphToGraphDef(graph, buffer.Handle, s.Handle);
                    s.Check();
                    return GraphDef.Parser.ParseFrom(buffer.DangerousMemoryBlock.Stream());
                }
            }
        }

        public static FunctionDef GetFunctionDef(IntPtr func)
        {
            using var s = new Status();
            using var buffer = new Buffer();
            c_api.TF_FunctionToFunctionDef(func, buffer.Handle, s.Handle);
            s.Check(true);
            var func_def = FunctionDef.Parser.ParseFrom(buffer.ToArray());
            return func_def;
        }

        public static bool IsAddN(NodeDef node_def, int n)
        {
            if (node_def.Op != "AddN" || node_def.Name != "add" ||
                node_def.Input.Count != n)
            {
                return false;
            }

            bool found_t = false;
            bool found_n = false;
            foreach (var attr in node_def.Attr)
            {
                if (attr.Key == "T")
                {
                    if (attr.Value.Type == DataType.DtInt32)
                    {
                        found_t = true;
                    }
                    else
                    {
                        return false;
                    }
                }
                else if (attr.Key == "N")
                {
                    if (attr.Value.I == n)
                    {
                        found_n = true;
                    }
                    else
                    {
                        return false;
                    }
                }
            }

            return found_t && found_n;
        }

        public static bool IsNeg(NodeDef node_def, string input)
        {
            return node_def.Op == "Neg" && node_def.Name == "neg" &&
                   node_def.Input.Count == 1 && node_def.Input[0] == input;
        }

        public static bool IsPlaceholder(NodeDef node_def)
        {
            if (node_def.Op != "Placeholder" || node_def.Name != "feed")
            {
                return false;
            }

            bool found_dtype = false;
            bool found_shape = false;
            foreach (var attr in node_def.Attr)
            {
                if (attr.Key == "dtype")
                {
                    if (attr.Value.Type == DataType.DtInt32)
                    {
                        found_dtype = true;
                    }
                    else
                    {
                        return false;
                    }
                }
                else if (attr.Key == "shape")
                {
                    found_shape = true;
                }
            }

            return found_dtype && found_shape;
        }

        public static bool IsScalarConst(NodeDef node_def, int v)
        {
            if (node_def.Op != "Const" || node_def.Name != "scalar")
            {
                return false;
            }

            bool found_dtype = false;
            bool found_value = false;
            foreach (var attr in node_def.Attr)
            {
                if (attr.Key == "dtype")
                {
                    if (attr.Value.Type == DataType.DtInt32)
                    {
                        found_dtype = true;
                    }
                    else
                    {
                        return false;
                    }
                }
                else if (attr.Key == "value")
                {
                    if (attr.Value.Tensor != null &&
                        attr.Value.Tensor.IntVal.Count == 1 &&
                        attr.Value.Tensor.IntVal[0] == v)
                    {
                        found_value = true;
                    }
                    else
                    {
                        return false;
                    }
                }
            }

            return found_dtype && found_value;
        }

        public static Operation Neg(Operation n, Graph graph, Status s, string name = "neg")
        {
            lock (Locks.ProcessWide)
            {
                OperationDescription desc = c_api.TF_NewOperation(graph, "Neg", name);
                var neg_input = new TF_Output(n, 0);
                c_api.TF_AddInput(desc, neg_input);
                var op = c_api.TF_FinishOperation(desc, s.Handle);
                s.Check();

                return op;
            }
        }

        public static Operation Placeholder(Graph graph, Status s, string name = "feed", TF_DataType dtype = TF_DataType.TF_INT32, long[] dims = null)
        {
            lock (Locks.ProcessWide)
            {
                var desc = c_api.TF_NewOperation(graph, "Placeholder", name);
                c_api.TF_SetAttrType(desc, "dtype", dtype);
                if (dims != null)
                {
                    c_api.TF_SetAttrShape(desc, "shape", dims, dims.Length);
                }

                var op = c_api.TF_FinishOperation(desc, s.Handle);
                s.Check();

                return op;
            }
        }

        public static Operation Const(Tensor t, Graph graph, Status s, string name)
        {
            lock (Locks.ProcessWide)
            {
                var desc = c_api.TF_NewOperation(graph, "Const", name);
                c_api.TF_SetAttrTensor(desc, "value", t, s.Handle);
                s.Check();
                c_api.TF_SetAttrType(desc, "dtype", t.dtype);
                var op = c_api.TF_FinishOperation(desc, s.Handle);
                s.Check();

                return op;
            }
        }

        public static Operation ScalarConst(int v, Graph graph, Status s, string name = "scalar")
        {
            return Const(new Tensor(v), graph, s, name);
        }

        public static Tensor Int32Tensor(int v)
        {
            return new Tensor(v);
        }
    }
}