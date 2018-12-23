using System;
using System.Collections.Generic;
using System.IO;
using System.Text;

namespace Tensorflow
{
    public static class gen_math_ops
    {
        public static OpDefLibrary _op_def_lib = _InitOpDefLibrary();

        public static Tensor add(Tensor a, Tensor b)
        {
            var keywords = new Dictionary<string, object>();
            keywords.Add("x", a);
            keywords.Add("y", b);

            var _op = _op_def_lib._apply_op_helper("Add", name: "add", keywords: keywords);

            var tensor = new Tensor(_op, 0, TF_DataType.TF_FLOAT);

            return tensor;
        }

        private static OpDefLibrary _InitOpDefLibrary()
        {
            // c_api.TF_GraphGetOpDef(g.Handle, op_type_name, buffer.Handle, status.Handle);
            var bytes = File.ReadAllBytes("Tensorflow/op_list_proto_math.bin");
            var op_list = OpList.Parser.ParseFrom(bytes);
            var op_def_lib = new OpDefLibrary();
            op_def_lib.add_op_list(op_list);

            return op_def_lib;
        }
    }
}
