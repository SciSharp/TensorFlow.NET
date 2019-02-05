using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Dynamic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using static Tensorflow.OpDef.Types;

namespace Tensorflow
{
    public class OpDefLibrary
    {
        public Operation _apply_op_helper(string op_type_name, string name = "", dynamic args = null)
        {
            var keywords = ConvertToDict(args);
            var g = ops.get_default_graph();
            var op_def = g.GetOpDef(op_type_name);

            // Default name if not specified.
            if (String.IsNullOrEmpty(name))
                name = op_type_name;

            // Check for deprecation
            if (op_def.Deprecation != null && op_def.Deprecation.Version > 0)
            {

            }

            var default_type_attr_map = new Dictionary<string, object>();
            foreach (var attr_def in op_def.Attr)
            {
                if (attr_def.Type != "type") continue;
                var key = attr_def.Name;
                if (attr_def.DefaultValue != null)
                {
                    default_type_attr_map[key] = attr_def.DefaultValue.Type;
                }
            }

            var attrs = new Dictionary<string, object>();
            var inputs = new List<Tensor>();
            var input_types = new List<TF_DataType>();
            var base_types = new List<TF_DataType>();

            Operation op = null;
            Python.with<ops.name_scope>(new ops.name_scope(name), scope =>
            {
                // Perform input type inference
                foreach (var input_arg in op_def.InputArg)
                {
                    var input_name = input_arg.Name;
                    var values = keywords[input_name];
                    // Goals:
                    // * Convert values to Tensors if it contains constants.
                    // * Verify that values is a list if that matches the input_arg's
                    // type.
                    // * If the input_arg's type is determined by attrs, either set
                    // those attrs and validate those attr values are legal (if
                    // they have not yet been set) or validate the input matches
                    // the type indicated by the attrs (if they have already been
                    // inferred via an earlier input).
                    // * If the input_arg has an explicit type, make sure the input
                    // conforms.

                    if (_IsListParameter(input_arg))
                    {
                        DataType dtype = DataType.DtInvalid;
                        DataType default_dtype = DataType.DtInvalid;

                        if (!_IsListValue(values))
                            throw new TypeError($"Expected list for '{input_name}' argument to '{op_type_name}' Op, not {values}.");
                        if(input_arg.Type != DataType.DtInvalid)
                        {
                            dtype = input_arg.Type;
                        }
                        else if (!String.IsNullOrEmpty(input_arg.NumberAttr))
                        {

                        }

                        if(input_arg.IsRef && dtype != DataType.DtInvalid)
                            dtype = dtype.as_base_dtype();

                        values = ops.internal_convert_n_to_tensor(values, name: input_arg.Name, dtype: dtype, preferred_dtype: default_dtype, as_ref: input_arg.IsRef);
                        
                        inputs.AddRange(values as Tensor[]);
                    }
                    else
                    {
                        if (!(values is Tensor))
                        {
                            keywords[input_name] = constant_op.constant(values, input_name);
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

                            values = new Tensor[] { value };
                        }
                    }

                    base_types.AddRange((values as Tensor[]).Select(x => x.dtype.as_base_dtype()));
                    input_types.AddRange(base_types);
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
                    if (!attrs.ContainsKey(key))
                        Console.WriteLine($"{key} not found in attr_def.");
                    var value = attrs[key];
                    var attr_value = new AttrValue();

                    switch (attr_def.Type)
                    {
                        case "string":
                            attr_value.S = Google.Protobuf.ByteString.CopyFromUtf8((string)value);
                            break;
                        case "type":
                            attr_value.Type = _MakeType((TF_DataType)value, attr_def);
                            break;
                        case "bool":
                            attr_value.B = (bool)value;
                            break;
                        case "shape":
                            attr_value.Shape = value == null ?
                                attr_def.DefaultValue.Shape :
                                tensor_util.as_shape((long[])value);
                            break;
                        default:
                            throw new InvalidDataException($"attr_def.Type {attr_def.Type}");
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
                op = g.create_op(op_type_name, inputs, output_types.ToArray(),
                    name: scope,
                    input_types: input_types.ToArray(),
                    attrs: attr_protos,
                    op_def: op_def);
            });

            return op;
        }

        public DataType _MakeType(TF_DataType v, AttrDef attr_def)
        {
            return v.as_base_dtype().as_datatype_enum();
        }

        private bool _IsListParameter(ArgDef arg)
        {
            if (!String.IsNullOrEmpty(arg.NumberAttr))
                return true;
            else if (!String.IsNullOrEmpty(arg.TypeListAttr))
                return true;
            else
                return false;
        }

        private bool _IsListValue(object v)
        {
            switch (v)
            {
                case Tensor[] val:
                    return true;
                default:
                    return false;
            }
        }

        private Dictionary<string, object> ConvertToDict(dynamic dyn)
        {
            var dictionary = new Dictionary<string, object>();
            foreach (PropertyDescriptor propertyDescriptor in TypeDescriptor.GetProperties(dyn))
            {
                object obj = propertyDescriptor.GetValue(dyn);
                string name = propertyDescriptor.Name;
                // avoid .net keyword
                if (name == "_ref_")
                    name = "ref";
                dictionary.Add(name, obj);
            }
            return dictionary;
        }
    }
}
