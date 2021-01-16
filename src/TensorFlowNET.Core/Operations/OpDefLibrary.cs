/*****************************************************************************
   Copyright 2018 The TensorFlow.NET Authors. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
******************************************************************************/

using Google.Protobuf;
using System;
using System.Collections.Generic;
using System.Linq;
using static Tensorflow.Binding;
using static Tensorflow.OpDef.Types;

namespace Tensorflow
{
    public class OpDefLibrary
    {
        public Operation _apply_op_helper(string op_type_name, string name = null, object args = null)
            => _apply_op_helper(op_type_name, name: name, keywords: ConvertToDict(args));

        public Operation _apply_op_helper(string op_type_name, string name = null, Dictionary<string, object> keywords = null)
        {
            var g = ops._get_graph_from_inputs(keywords == null ? new object[0] : keywords.Values.ToArray());
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
            object values = null;

            g.as_default();
            var ret_op = tf_with(ops.name_scope(name), scope =>
            {
                var inferred_from = new Dictionary<string, object>();
                var base_types = new List<TF_DataType>();
                var types = new List<TF_DataType>();
                string _scope_name = scope;

                // Perform input type inference
                foreach (var input_arg in op_def.InputArg)
                {
                    var input_name = input_arg.Name;

                    if (keywords.ContainsKey(input_name))
                        values = keywords[input_name];
                    else if (keywords.ContainsKey(input_name + "_"))
                    {
                        input_name += "_";
                        values = keywords[input_name];
                    }
                    else
                        throw new TypeError("No argument for input " + input_name);

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

                    DataType dtype = DataType.DtInvalid;
                    DataType default_dtype = DataType.DtInvalid;

                    if (_IsListParameter(input_arg))
                    {
                        if (!_IsListValue(values))
                            throw new TypeError($"Expected list for '{input_name}' argument to '{op_type_name}' Op, not {values}.");
                        if (input_arg.Type != DataType.DtInvalid)
                            dtype = input_arg.Type;
                        else if (!String.IsNullOrEmpty(input_arg.NumberAttr))
                        {
                            if (attrs.ContainsKey(input_arg.TypeAttr))
                                dtype = (DataType)attrs[input_arg.TypeAttr];
                            else
                                switch (values)
                                {
                                    case Tensor[] values1:
                                        dtype = values1[0].dtype.as_datatype_enum();
                                        break;
                                    case object[] values1:
                                        foreach (var t in values1)
                                            if (t is Tensor tensor)
                                            {
                                                dtype = tensor.dtype.as_datatype_enum();
                                                break;
                                            }
                                        break;
                                    default:
                                        throw new NotImplementedException($"can't infer the dtype for {values.GetType()}");
                                }

                            if (dtype == DataType.DtInvalid && default_type_attr_map.ContainsKey(input_arg.TypeAttr))
                                default_dtype = (DataType)default_type_attr_map[input_arg.TypeAttr];
                        }

                        if (!input_arg.IsRef && dtype != DataType.DtInvalid)
                            dtype = dtype.as_base_dtype();

                        values = ops.internal_convert_n_to_tensor(values as object[],
                            name: input_arg.Name,
                            dtype: dtype.as_tf_dtype(),
                            preferred_dtype: default_dtype.as_tf_dtype(),
                            as_ref: input_arg.IsRef);
                    }
                    else
                    {
                        if (input_arg.Type != DataType.DtInvalid)
                            dtype = input_arg.Type;
                        else if (attrs.ContainsKey(input_arg.TypeAttr))
                            dtype = (DataType)attrs[input_arg.TypeAttr];
                        else if (isinstance(values, typeof(string)) && dtype == DataType.DtInvalid)
                            dtype = DataType.DtString;
                        else if (default_type_attr_map.ContainsKey(input_arg.TypeAttr))
                            default_dtype = (DataType)default_type_attr_map[input_arg.TypeAttr];

                        var value = ops.convert_to_tensor(values,
                            name: input_name,
                            dtype: dtype.as_tf_dtype(),
                            as_ref: input_arg.IsRef,
                            preferred_dtype: default_dtype.as_tf_dtype());

                        //if (!String.IsNullOrEmpty(input_arg.TypeAttr))
                        //attrs[input_arg.TypeAttr] = values.dtype;

                        values = new Tensor[] { value };
                    }

                    if (values is Tensor[] values2)
                    {
                        types = values2.Select(x => x.dtype).ToList();
                        inputs.AddRange(values2);
                        base_types = values2.Select(x => x.dtype.as_base_dtype()).ToList();
                    }
                    else throw new NotImplementedException("_IsListParameter");

                    SetAttrs(op_type_name,
                        input_arg,
                        op_def,
                        attrs,
                        inferred_from,
                        types,
                        base_types,
                        input_types,
                        values);
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
                foreach (AttrDef attr_def in op_def.Attr)
                {
                    var key = attr_def.Name;
                    if (attrs.ContainsKey(key))
                    {
                        attr_protos[key] = SetAttrValue(op_def, attr_def, attrs[key]);
                    }
                    else
                    {
                        if (attr_def.DefaultValue == null)
                        {
                            throw new TypeError("Missing required positional argument " + key);
                        }
                    }
                }

                attrs.Clear();

                // Determine output types (possibly using attrs)
                var output_types = new List<TF_DataType>();

                foreach (var arg in op_def.OutputArg)
                {
                    types = new List<TF_DataType>();
                    if (!string.IsNullOrEmpty(arg.NumberAttr))
                    {

                    }
                    else if (!string.IsNullOrEmpty(arg.TypeAttr))
                    {
                        types = new List<TF_DataType>() { (TF_DataType)attr_protos[arg.TypeAttr].Type };
                    }

                    if (arg.IsRef)
                        types = types.Select(x => x.as_ref()).ToList();

                    output_types.AddRange(types);
                }

                // We add an explicit colocation constraint between
                // the newly created op and any of its reference-typed inputs.
                var must_colocate_inputs = zip(op_def.InputArg, inputs)
                    .Where(x => x.Item1.IsRef)
                    .Select(x => x.Item2)
                    .ToArray();

                _MaybeColocateWith(must_colocate_inputs);

                // Add Op to graph
                var op = g.create_op(op_type_name,
                    inputs.ToArray(),
                    output_types.ToArray(),
                    name: _scope_name,
                    input_types: input_types.ToArray(),
                    attrs: attr_protos,
                    op_def: op_def);

                return op;
            });
            g.Exit();
            return ret_op;
        }

        private void _MaybeColocateWith(ITensorOrOperation[] inputs)
        {

        }

        private void SetAttrs(string op_type_name,
            ArgDef input_arg,
            OpDef op_def,
            Dictionary<string, object> attrs,
            Dictionary<string, object> inferred_from,
            List<TF_DataType> types,
            List<TF_DataType> base_types,
            List<TF_DataType> input_types,
            object values)
        {
            var input_name = input_arg.Name;

            if (!string.IsNullOrEmpty(input_arg.NumberAttr))
            {
                if (attrs.ContainsKey(input_arg.NumberAttr))
                {

                }
                else
                {
                    if(values is Tensor[] tensors)
                    {
                        var num_attr = op_def.Attr.First(x => x.Name == input_arg.NumberAttr);
                        if (num_attr.HasMinimum && tensors.Length < num_attr.Minimum)
                            throw new ValueError($"List argument '{input_name}' to '{op_type_name}' Op with length {(values as Tensor[]).Length} shorter " +
                                $"than minimum length {num_attr.Minimum}");

                        attrs[input_arg.NumberAttr] = Convert.ToInt64(tensors.Length);
                        inferred_from[input_arg.NumberAttr] = input_name;
                    }
                }

                // All tensors must have the same base type.
                if (input_arg.Type != DataType.DtInvalid)
                {

                }
                else
                {
                    attrs[input_arg.TypeAttr] = base_types[0];
                    inferred_from[input_arg.TypeAttr] = input_name;
                    var type_attr = op_def.Attr.First(x => x.Name == input_arg.TypeAttr);
                }
            }
            else if (!string.IsNullOrEmpty(input_arg.TypeAttr))
            {
                var attr_value = base_types[0];
                if (attrs.ContainsKey(input_arg.TypeAttr))
                {

                }
                else
                {
                    attrs[input_arg.TypeAttr] = attr_value;
                    inferred_from[input_arg.TypeAttr] = input_name;
                }
            }
            else if (!string.IsNullOrEmpty(input_arg.TypeListAttr))
            {
                var attr_value = base_types;
                if (attrs.ContainsKey(input_arg.TypeListAttr))
                {

                }
                else
                {
                    attrs[input_arg.TypeListAttr] = attr_value;
                    inferred_from[input_arg.TypeListAttr] = input_name;
                }
            }

            if (input_arg.IsRef)
                input_types.AddRange(types);
            else
                input_types.AddRange(base_types);
        }

        public ByteString _MakeStr(string value, AttrDef attr_def)
        {
            return ByteString.CopyFromUtf8(value ?? string.Empty);
        }

        public TensorShapeProto _MakeShape(TensorShape shape, AttrDef attr_def)
        {
            return shape.as_proto();
        }

        public DataType _MakeType(TF_DataType v, AttrDef attr_def)
        {
            return v.as_base_dtype().as_datatype_enum();
        }

        private AttrValue SetAttrValue(OpDef op_def, AttrDef attr_def, object value)
        {
            var attr_value = new AttrValue();

            if (attr_def.Type.StartsWith("list("))
            {
                if (attr_def.HasMinimum)
#pragma warning disable CS0642 // Possible mistaken empty statement
                    ;
#pragma warning restore CS0642 // Possible mistaken empty statement
                attr_value.List = new AttrValue.Types.ListValue();
            }

            switch (attr_def.Type)
            {
                case "string":
                    attr_value.S = _MakeStr((string)value, attr_def);
                    break;
                case "type":
                    attr_value.Type = _MakeType((TF_DataType)value, attr_def);
                    break;
                case "list(type)":
                    attr_value.List.Type.AddRange((value as IList<TF_DataType>).Select(x => _MakeType(x, attr_def)));
                    break;
                case "list(int)":
                    attr_value.List.I.AddRange((value as int[]).Select(x => Convert.ToInt64(x)));
                    break;
                case "bool":
                    attr_value.B = (bool)value;
                    break;
                case "float":
                    attr_value.F = (float)value;
                    break;
                case "int":
                    if (value is long value_long)
                        attr_value.I = value_long;
                    else
                        attr_value.I = Convert.ToInt64(value);
                    if (attr_def.HasMinimum && attr_value.I < attr_def.Minimum)
                        throw new ValueError($"Attr '{attr_def.Name}' of '{op_def.Name}' Op passed {attr_value.I} less than minimum {attr_def.Minimum}.");
                    break;
                case "shape":
                    if (value == null && attr_def.DefaultValue != null)
                        attr_value.Shape = attr_def.DefaultValue.Shape;

                    if (value is TensorShape val1)
                        attr_value.Shape = val1.as_proto();
                    else if (value is long[] val2)
                        attr_value.Shape = tensor_util.as_shape(val2);
                    else if (value is int[] val3)
                        attr_value.Shape = tensor_util.as_shape(val3);

                    break;
                case "list(shape)":
                    attr_value.List.Shape.AddRange((value as TensorShape[]).Select(x => _MakeShape(x, attr_def)));
                    break;
                default:
                    throw new TypeError($"SetAttrValue: can't not convert attr_def.Type '{attr_def.Type}' to protos.");
            }

            return attr_value;
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
            return v.GetType().IsArray;
        }
    }
}
