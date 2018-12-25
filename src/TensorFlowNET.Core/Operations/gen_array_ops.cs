using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using Tensorflow;

namespace Tensorflow
{
    public static class gen_array_ops
    {
        public static OpDefLibrary _op_def_lib = _InitOpDefLibrary();

        public static Tensor placeholder(TF_DataType dtype, TensorShape shape = null)
        {
            /*var g = ops.get_default_graph();
            var op = new Operation(g, "Placeholder", "feed");

            var tensor = new Tensor(op, 0, dtype);

            return tensor;*/

            var keywords = new Dictionary<string, object>();
            keywords.Add("dtype", dtype);
            keywords.Add("shape", shape);

            var _op = _op_def_lib._apply_op_helper("Placeholder", keywords: keywords);
            var _result = _op.outputs;
            var _inputs_flat = _op.inputs;
            var _attrs = new Dictionary<string, object>();

            _attrs["dtype"] = _op.get_attr("dtype");
            _attrs["shape"] = _op.get_attr("shape");

            var tensor = new Tensor(_op, 0, dtype);
            return tensor;
        }

        private static OpDefLibrary _InitOpDefLibrary()
        {
            // c_api.TF_GraphGetOpDef(g.Handle, op_type_name, buffer.Handle, status.Handle);
            var bytes = File.ReadAllBytes("Tensorflow/op_list_proto_array.bin");
            var op_list = OpList.Parser.ParseFrom(bytes);
            var op_def_lib = new OpDefLibrary();
            op_def_lib.add_op_list(op_list);

            return op_def_lib;
        }
    }
}
