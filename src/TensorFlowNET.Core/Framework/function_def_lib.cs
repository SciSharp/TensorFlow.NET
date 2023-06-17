using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Security.Cryptography;
using System.Text;
using Tensorflow.Graphs;
using Tensorflow.Common.Extensions;
using static Tensorflow.Binding;
using static Tensorflow.CppShapeInferenceResult.Types;

namespace Tensorflow.Framework
{
    public class function_def_lib
    {
        // TODO(Rinne): process signatures and structured outputs.
        public static FuncGraph function_def_to_graph(FunctionDef fdef, object? structured_input_signature, 
            object? structured_outputs, List<TensorShapeProto> input_shapes = null)
        {
            var func_graph = new FuncGraph(fdef.Signature.Name);
            if(input_shapes is null)
            {
                if(fdef.Attr.TryGetValue("_input_shapes", out var input_shapes_attr))
                {
                    var raw_input_shapes = input_shapes_attr.List.Shape;
                    input_shapes = new List<TensorShapeProto>();
                    foreach(var (input_shape, arg_def) in raw_input_shapes.Zip(fdef.Signature.InputArg, (x, y) => (x, y)))
                    {
                        if(arg_def.Type == DataType.DtResource && arg_def.HandleData is not null && arg_def.HandleData.Count > 0)
                        {
                            input_shapes.Add(null);
                        }
                        else
                        {
                            input_shapes.Add(input_shape);
                        }
                    }
                }
            }

            var (graph_def, nested_to_flat_tensor_name) = function_def_to_graph_def(fdef, input_shapes);

            func_graph.as_default();
            importer.import_graph_def(graph_def, name: "", validate_colocation_constraints: false);
            var input_tensor_names = fdef.Signature.InputArg.Select(x => nested_to_flat_tensor_name[x.Name]);
            func_graph.Inputs = new Tensors(input_tensor_names.Select(x => func_graph.get_tensor_by_name(x)).ToArray());

            var output_tensor_names = fdef.Signature.OutputArg.Select(x => nested_to_flat_tensor_name[fdef.Ret[x.Name]]);
            func_graph.Outputs = new Tensors(output_tensor_names.Select(x => func_graph.get_tensor_by_name(x)).ToArray());
            // TODO(Rinne): func_graph.ControlOutputs
            _set_handle_data(func_graph, fdef);

            foreach(var node in graph_def.Node)
            {
                if(node.Attr.TryGetValue("_output_shapes", out var output_shapes))
                {
                    var op = func_graph.get_operation_by_name(node.Name);
                    foreach(var (output_index, shape) in enumerate(output_shapes.List.Shape.Take(op.outputs.Length)))
                    {
                        op.outputs[output_index].shape = new Shape(shape);
                    }
                }
            }
            Dictionary<long, string> output_names = new();
            foreach(var (ret_arg_def, tensor_name) in zip(fdef.Signature.OutputArg, output_tensor_names))
            {
                output_names[ops.tensor_id(func_graph.get_tensor_by_name(tensor_name))] = ret_arg_def.Name;
            }
            func_graph._output_names = output_names;

            func_graph.Exit();
            return func_graph;
        }

        public static (GraphDef, Dictionary<string, string>) function_def_to_graph_def(FunctionDef fdef, List<TensorShapeProto> input_shapes)
        {
            var graph_def = new GraphDef()
            {
                Versions = new VersionDef()
                {
                    Producer = versions.GRAPH_DEF_VERSION,
                    MinConsumer = versions.GRAPH_DEF_VERSION_MIN_CONSUMER
                }
            };

            var default_graph = ops.get_default_graph();

            if(input_shapes is not null && input_shapes.Count > 0 && input_shapes.Count != fdef.Signature.InputArg.Count)
            {
                throw new ValueError($"Length of `input_shapes` must match the number " +
                    $"of `input_arg`s in `fdef`. Got {input_shapes.Count} `input_shapes` and " +
                    $"{fdef.Signature.InputArg.Count} `input_arg`s.");
            }

            foreach(var (i, arg_def) in enumerate(fdef.Signature.InputArg))
            {
                NodeDef node_def = new();
                node_def.Name = arg_def.Name;
                node_def.Op = "Placeholder";
                node_def.Attr["dtype"] = new AttrValue()
                {
                    Type = arg_def.Type
                };
                if(input_shapes is not null && input_shapes.Count > 0 && input_shapes[i] is not null)
                {
                    var input_shape = input_shapes[i];
                    // skip the condition that input_shape is not `TensorShapeProto`.
                    AttrValue shape = new AttrValue()
                    {
                        Shape = new TensorShapeProto()
                    };
                    shape.Shape = new TensorShapeProto(input_shape);
                    node_def.Attr["shape"] = shape;
                }
                if (!fdef.ArgAttr.ContainsKey((uint)i))
                {
                    fdef.ArgAttr[(uint)i] = new FunctionDef.Types.ArgAttrs();
                }
                var arg_attrs = fdef.ArgAttr[(uint)i].Attr;
                foreach(var k in arg_attrs.Keys)
                {
                    if(k == "_output_shapes")
                    {
                        if (arg_attrs[k].ValueCase == AttrValue.ValueOneofCase.List)
                        {
                            node_def.Attr["shape"].Shape = new TensorShapeProto(arg_attrs[k].List.Shape[0]);
                        }
                        else if (arg_attrs[k].ValueCase == AttrValue.ValueOneofCase.Shape)
                        {
                            node_def.Attr["shape"].Shape = new TensorShapeProto(arg_attrs[k].Shape);
                        }
                    }
                    else if (k.StartsWith("_"))
                    {
                        if (!node_def.Attr.ContainsKey(k))
                        {
                            node_def.Attr[k] = new AttrValue();
                        }
                        node_def.Attr[k] = new AttrValue(arg_attrs[k]);
                    }
                }

                graph_def.Node.Add(node_def);
            }

            graph_def.Node.AddRange(fdef.NodeDef);

            Dictionary<string, string> nested_to_flat_tensor_name = new();
            foreach(var arg_def in fdef.Signature.InputArg)
            {
                nested_to_flat_tensor_name[arg_def.Name] = $"{arg_def.Name}:0";
                string control_name = "^" + arg_def.Name;
                nested_to_flat_tensor_name[control_name] = control_name;
            }

            foreach(var node_def in fdef.NodeDef)
            {
                var graph = default_graph;
                while (true)
                {
                    if(graph is null)
                    {
                        break;
                    }
                    var f = graph.Functions.GetOrDefault(node_def.Op, null);
                    if(f is not null && graph.OuterGraph is null)
                    {
                        break;
                    }
                    graph = graph.OuterGraph;
                }

                var op_def = default_graph.GetOpDef(node_def.Op);

                foreach(var attr in op_def.Attr)
                {
                    if(attr.Type == "func")
                    {
                        var fname = node_def.Attr[attr.Name].Func.Name;
                        if (!is_function(fname))
                        {
                            throw new ValueError($"Function {fname} was not found. Please make sure " +
                                $"the FunctionDef `fdef` is correct.");
                        }
                    }
                    else if(attr.Type == "list(func)")
                    {
                        foreach(var fn in node_def.Attr[attr.Name].List.Func)
                        {
                            var fname = fn.Name;
                            if (!is_function(fname))
                            {
                                throw new ValueError($"Function {fname} was not found. Please make " +
                                    $"sure the FunctionDef `fdef` is correct.");
                            }
                        }
                    }
                }

                int flattened_index = 0;
                foreach(var arg_def in op_def.OutputArg)
                {
                    var num_args = _get_num_args(arg_def, node_def);
                    for(int i = 0; i < num_args; i++)
                    {
                        var nested_name = $"{node_def.Name}:{arg_def.Name}:{i}";
                        var flat_name = $"{node_def.Name}:{flattened_index}";
                        nested_to_flat_tensor_name[nested_name] = flat_name;
                        flattened_index++;
                    }
                }
                string control_name = "^" + node_def.Name;
                nested_to_flat_tensor_name[control_name] = control_name;
            }

            foreach(var node_def in graph_def.Node)
            {
                for(int i = 0; i < node_def.Input.Count; i++)
                {
                    node_def.Input[i] = nested_to_flat_tensor_name[node_def.Input[i]];
                }
            }

            return (graph_def, nested_to_flat_tensor_name);
        }

        private static void _set_handle_data(FuncGraph func_graph, FunctionDef fdef)
        {
            foreach(var (tensor, arg_def) in zip(func_graph.Inputs, fdef.Signature.InputArg).Concat(zip(func_graph.Outputs, fdef.Signature.OutputArg)))
            {
                if(arg_def.HandleData is not null && arg_def.HandleData.Count > 0)
                {
                    tensor.shape = Shape.Scalar;

                    var shape_and_type = arg_def.HandleData[0];
                    var handle_data = new HandleData();
                    handle_data.IsSet = true;
                    handle_data.ShapeAndType.Add(new HandleShapeAndType()
                    {
                        Shape = shape_and_type.Shape,
                        Dtype = shape_and_type.Dtype
                    });
                    resource_variable_ops._set_handle_shapes_and_types(tensor, handle_data, true);
                }
            }
        }

        private static long _get_num_args(OpDef.Types.ArgDef arg_def, NodeDef node_def)
        {
            if (!string.IsNullOrEmpty(arg_def.NumberAttr))
            {
                return node_def.Attr[arg_def.NumberAttr].I;
            }
            else if(!string.IsNullOrEmpty(arg_def.TypeListAttr))
            {
                return node_def.Attr[arg_def.TypeListAttr].List.Type.Count;
            }
            else if(arg_def.TypeAttr is not null || arg_def.Type != DataType.DtInvalid)
            {
                return 1;
            }
            else
            {
                throw new ValueError($"Invalid arg_def:\n\n{arg_def}. Please make sure the " +
                    $"FunctionDef `fdef` is correct.");
            }
        }

        public static bool is_function(string fname)
        {
            if (tf.Context.executing_eagerly())
            {
                return tf.Context.has_function(fname);
            }
            else
            {
                var graph = ops.get_default_graph();
                while(graph is not null)
                {
                    if (graph.IsFunction(fname))
                    {
                        return true;
                    }
                    if(graph.OuterGraph is not null)
                    {
                        graph = graph.OuterGraph;
                    }
                    else
                    {
                        return false;
                    }
                }
            }
            throw new ValueError("Unexpected behavior happened in runtime, please submit an issue to " +
                "https://github.com/SciSharp/TensorFlow.NET/issues");
        }
    }
}
