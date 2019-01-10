using NumSharp.Core;
using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;
using static Tensorflow.OpDef.Types;

namespace Tensorflow
{
    public class OpDefLibrary
    {
        public Operation _apply_op_helper(string op_type_name, string name = "", Dictionary<string, object> keywords = null)
        {
            var g = ops.get_default_graph();
            var op_def = g.GetOpDef(op_type_name);

            if (String.IsNullOrEmpty(name))
            {
                name = op_type_name;
            }

            string scope = g.unique_name(name) + "/";

            var default_type_attr_map = new Dictionary<string, object>();
            foreach (var attr_def in op_def.Attr)
            {
                if (attr_def.Type != "type") continue;
                var key = attr_def.Name;
                if(attr_def.DefaultValue != null)
                {
                    default_type_attr_map[key] = attr_def.DefaultValue.Type;
                }
            }

            var attrs = new Dictionary<string, object>();
            var inputs = new List<Tensor>();
            var input_types = new List<TF_DataType>();

            // Perform input type inference
            foreach (var input_arg in op_def.InputArg)
            {
                var input_name = input_arg.Name;
                if (keywords[input_name] is double int_value)
                {
                    keywords[input_name] = constant_op.Constant(int_value, input_name);
                }

                if (keywords[input_name] is Tensor value)
                {
                    if (keywords.ContainsKey(input_name))
                    {
                        inputs.Add(value);
                    }

                    if (!String.IsNullOrEmpty(input_arg.TypeAttr))
                    {
                        attrs[input_arg.TypeAttr] = value.dtype;
                    }

                    if (input_arg.IsRef)
                    {

                    }
                    else
                    {
                        input_types.Add(value.dtype);
                    }
                }
            }

            // Process remaining attrs
            foreach (var attr in op_def.Attr)
            {
                if (keywords.ContainsKey(attr.Name))
                {
                    attrs[attr.Name] = keywords[attr.Name];
                }
            }

            // Convert attr values to AttrValue protos.
            var attr_protos = new Dictionary<string, AttrValue>();
            foreach (var attr_def in op_def.Attr)
            {
                var key = attr_def.Name;
                var value = attrs[key];
                var attr_value = new AttrValue();
                
                switch (attr_def.Type)
                {
                    case "type":
                        attr_value.Type = _MakeType((TF_DataType)value, attr_def);
                        break;
                    case "bool":
                        attr_value.B = (bool)value;
                        break;
                    case "shape":
                        attr_value.Shape = new TensorShapeProto();
                        break;
                }

                attr_protos[key] = attr_value;
            }

            // Determine output types (possibly using attrs)
            var output_types = new List<TF_DataType>();

            foreach (var arg in op_def.OutputArg)
            {
                if (!String.IsNullOrEmpty(arg.NumberAttr))
                {

                }
                else if (!String.IsNullOrEmpty(arg.TypeAttr))
                {
                    output_types.Add((TF_DataType)attr_protos[arg.TypeAttr].Type);
                }
            }

            // Add Op to graph
            var op = g.create_op(op_type_name, inputs, output_types.ToArray(),
                name: scope,
                input_types: input_types.ToArray(),
                attrs: attr_protos,
                op_def: op_def);

            return op;
        }

        public DataType _MakeType(TF_DataType v, AttrDef attr_def)
        {
            return v.as_datatype_enum();
        }
    }
}
