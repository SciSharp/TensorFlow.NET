using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;
using Tensorflow;
using static Tensorflow.OpDef.Types;

namespace TensorFlowNET.Core
{
    public class OpDefLibrary
    {
        public Dictionary<string, OpDef> _ops = new Dictionary<string, OpDef>();

        public void add_op_list(OpList op_list)
        {
            foreach(var op_def in op_list.Op)
            {
                add_op(op_def);
            }
        }

        public void add_op(OpDef op_def)
        {
            _ops[op_def.Name] = op_def;
        }

        public unsafe Operation _apply_op_helper(string op_type_name, string name = "", DataType? dtype = null, TensorShape shape = null)
        {
            var op_def = _ops[op_type_name];

            var status = new Status();
            var buffer = new Buffer();

            var g = ops.get_default_graph();

            if (String.IsNullOrEmpty(name))
            {
                name = op_type_name;
            }

            foreach(var attr_def in op_def.Attr)
            {
                if (attr_def.Type != "type") continue;
                var key = attr_def.Name;
            }

            foreach(var input_arg in op_def.InputArg)
            {

            }

            var attr_protos = new Dictionary<string, AttrValue>();
            foreach (var attr_def in op_def.Attr)
            {
                var key = attr_def.Name;
                var attr_value = new AttrValue();
                
                switch (attr_def.Type)
                {
                    case "type":
                        attr_value.Type = dtype.Value;
                        break;
                    case "shape":
                        attr_value.Shape = new TensorShapeProto();
                        break;
                }

                attr_protos[key] = attr_value;
            }

            var output_types = new List<DataType>();

            foreach (var arg in op_def.OutputArg)
            {
                if (!String.IsNullOrEmpty(arg.NumberAttr))
                {

                }
                else if (!String.IsNullOrEmpty(arg.TypeAttr))
                {
                    output_types.Add(attr_protos[arg.TypeAttr].Type);
                }
            }

            var op = g.create_op(op_type_name, null, output_types.ToArray(),
                name: "Placeholder_1/",
                input_types: new DataType[] { },
                attrs: null,
                op_def: null);

            return op;
        }
    }
}
