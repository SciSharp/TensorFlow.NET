using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using Tensorflow;

namespace TensorFlowNET.Core
{
    public static class gen_array_ops
    {
        public static OpDefLibrary _op_def_lib => _InitOpDefLibrary();

        public static Tensor placeholder(DataType dtype, TensorShape shape = null)
        {
            var _op = _op_def_lib._apply_op_helper("Placeholder", dtype: dtype, shape: shape);
            var _result = _op.outputs;
            var _inputs_flat = _op.inputs;
            var _attrs = new Dictionary<string, object>();

            _attrs["dtype"] = _op.get_attr("dtype");
            _attrs["shape"] = _op.get_attr("shape");

            return null;
        }

        private static OpDefLibrary _InitOpDefLibrary()
        {
            // c_api.TF_GraphGetOpDef(g.Handle, op_type_name, buffer.Handle, status.Handle);
            var bytes = File.ReadAllBytes("Tensorflow/op_list_proto_bytes.bin");
            var op_list = OpList.Parser.ParseFrom(bytes);
            var op_def_lib = new OpDefLibrary();
            op_def_lib.add_op_list(op_list);

            return op_def_lib;
        }
    }
}
