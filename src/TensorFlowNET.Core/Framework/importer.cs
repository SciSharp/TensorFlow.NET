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
using System.Diagnostics;
using System.Linq;
using static Tensorflow.Binding;
using static Tensorflow.OpDef.Types;

namespace Tensorflow
{
    public class importer
    {
        public static ITensorOrOperation[] import_graph_def_for_function(GraphDef graph_def, string name = null)
        {
            return import_graph_def(graph_def, validate_colocation_constraints: false, name: name);
        }
        public static ITensorOrOperation[] import_graph_def(GraphDef graph_def,
            Dictionary<string, Tensor> input_map = null,
            string[] return_elements = null,
            bool validate_colocation_constraints = true,
            string name = null,
            OpList producer_op_list = null)
        {
            var op_dict = op_def_registry.get_registered_ops();

            graph_def = _ProcessGraphDefParam(graph_def, op_dict);
            input_map = _ProcessInputMapParam(input_map);
            return_elements = _ProcessReturnElementsParam(return_elements);

            if (producer_op_list != null)
                _RemoveDefaultAttrs(op_dict, producer_op_list, graph_def);

            string prefix = "";
            var graph = ops.get_default_graph();
            tf_with(ops.name_scope(name, "import", input_map.Values), scope =>
            {
                prefix = scope;
                /*if (!string.IsNullOrEmpty(prefix))
                    prefix = prefix.Substring(0, prefix.Length - 1);
                else
                    prefix = "";*/

                // Generate any input map tensors inside name scope
                input_map = _ConvertInputMapValues(name, input_map);
            });

            TF_ImportGraphDefResults results = null;
            var bytes = graph_def.ToByteString().ToArray();
            var buffer = c_api_util.tf_buffer(bytes);
            var scoped_options = c_api_util.ScopedTFImportGraphDefOptions();
            var status = new Status();
            
            _PopulateTFImportGraphDefOptions(scoped_options, prefix, input_map, return_elements, validate_colocation_constraints );
            // need to create a class ImportGraphDefWithResults with IDisposal
            results = new TF_ImportGraphDefResults(c_api.TF_GraphImportGraphDefWithResults(graph, buffer, scoped_options, status));
            status.Check(true);

            _ProcessNewOps(graph);

            if (return_elements == null)
                return null;
            else
                return _GatherReturnElements(return_elements, graph, results);
        }

        private static ITensorOrOperation[] _GatherReturnElements(string[] requested_return_elements,
            Graph graph,
            TF_ImportGraphDefResults results)
        {
            var return_outputs = results.return_tensors;
            var return_opers = results.return_opers;

            var combined_return_elements = new List<ITensorOrOperation>();
            int outputs_idx = 0;
#pragma warning disable CS0219 // Variable is assigned but its value is never used
            int opers_idx = 0;
#pragma warning restore CS0219 // Variable is assigned but its value is never used
            foreach (var name in requested_return_elements)
            {
                if (name.Contains(":"))
                {
                    combined_return_elements.append(graph.get_tensor_by_tf_output(return_outputs[outputs_idx]));
                    outputs_idx += 1;
                }
                else
                {
                    throw new NotImplementedException("_GatherReturnElements");
                    // combined_return_elements.append(graph._get_operation_by_tf_operation(return_opers[opers_idx]));
                }
            }

            return combined_return_elements.ToArray();
        }

        private static void _ProcessNewOps(Graph graph)
        {
            foreach (var new_op in graph._add_new_tf_operations())
            {
                var original_device = new_op.Device;
                new_op._set_device(original_device);
            }
        }

        public static void _PopulateTFImportGraphDefOptions(ImportGraphDefOptions options,
            string prefix,
            Dictionary<string, Tensor> input_map,
            string[] return_elements, 
            bool validate_colocation_constraints)
        {
            c_api.TF_ImportGraphDefOptionsSetPrefix(options, prefix);
            c_api.TF_ImportGraphDefOptionsSetUniquifyNames(options.Options, true);

            foreach (var input in input_map)
            {
                var input_src = tf.compat.as_str(input.Key);
                var input_dst = input.Value;
                if (input_src.StartsWith("^"))
                {
                    var src_name = tf.compat.as_str(input_src.Substring(1));
                    var dst_op = input_dst._as_tf_output().oper;
                    c_api.TF_ImportGraphDefOptionsRemapControlDependency(options.Options, src_name, dst_op);
                }
                else
                {
                    var (src_name, src_index) = _ParseTensorName(input.Key);
                    src_name = tf.compat.as_str(src_name);
                    var dst_output = input_dst._as_tf_output();
                    c_api.TF_ImportGraphDefOptionsAddInputMapping(options.Options, src_name, src_index, dst_output);
                }
            }

            if (return_elements == null)
                return_elements = new string[0];

            foreach (var name in return_elements)
            {
                if (name.Contains(":"))
                {
                    var (op_name, index) = _ParseTensorName(name);
                    op_name = tf.compat.as_str(op_name);
                    c_api.TF_ImportGraphDefOptionsAddReturnOutput(options.Options, op_name, index);
                }
                else
                {
                    c_api.TF_ImportGraphDefOptionsAddReturnOperation(options.Options, tf.compat.as_str(name));
                }
            }

            c_api.TF_ImportGraphDefOptionsSetValidateColocationConstraints(options.Options, validate_colocation_constraints);
        }

        private static (string, int) _ParseTensorName(string tensor_name)
        {
            var components = tensor_name.Split(':');
            if (components.Length == 2)
                return (components[0], int.Parse(components[1]));
            else if (components.Length == 1)
                return (components[0], 0);
            else
                throw new ValueError($"Cannot convert {tensor_name} to a tensor name.");
        }

        public static Dictionary<string, Tensor> _ConvertInputMapValues(string name, Dictionary<string, Tensor> input_map)
        {
            return input_map;
        }

        public static GraphDef _ProcessGraphDefParam(GraphDef graph_def, Dictionary<string, OpDef> op_dict)
        {
            foreach (var node in graph_def.Node)
            {
                if (!op_dict.ContainsKey(node.Op))
                    continue;

                var op_def = op_dict[node.Op];
                _SetDefaultAttrValues(node, op_def);
            }

            return graph_def;
        }

        private static GraphDef _ProcessGraphDefParam(GraphDef graph_def)
        {
            var old_graph_def = graph_def;
            graph_def = new GraphDef(old_graph_def);

            return graph_def;
        }

        private static void _SetDefaultAttrValues(NodeDef node_def, OpDef op_def)
        {
            foreach (var attr_def in op_def.Attr)
            {
                var key = attr_def.Name;
                if (attr_def.DefaultValue != null)
                {
                    if (node_def.Attr.ContainsKey(key))
                    {
                        var value = node_def.Attr[key];
                        if (value == null)
                            node_def.Attr[key] = attr_def.DefaultValue;
                    }
                    else
                    {
                        node_def.Attr[key] = attr_def.DefaultValue;
                    }
                }
            }
        }

        private static Dictionary<string, Tensor> _ProcessInputMapParam(Dictionary<string, Tensor> input_map)
        {
            if (input_map == null)
                return new Dictionary<string, Tensor>();

            return input_map;
        }

        private static string[] _ProcessReturnElementsParam(string[] return_elements)
        {
            if (return_elements == null)
                return null;

            return return_elements;
        }

        private static void _RemoveDefaultAttrs(Dictionary<string, OpDef> op_dict, OpList producer_op_list, GraphDef graph_def)
        {
            var producer_op_dict = new Dictionary<string, OpDef>();
            producer_op_list.Op.Select(op =>
            {
                producer_op_dict[op.Name] = op;
                return op;
            }).ToArray();

            foreach (var node in graph_def.Node)
            {
                // Remove any default attr values that aren't in op_def.
                if (producer_op_dict.ContainsKey(node.Op))
                {
                    var op_def = op_dict[node.Op];
                    var producer_op_def = producer_op_dict[node.Op];
                    foreach (var key in node.Attr)
                    {
                        if (_FindAttrInOpDef(key.Key, op_def) == null)
                        {
                            var attr_def = _FindAttrInOpDef(key.Key, producer_op_def);
                            if (attr_def != null && attr_def.DefaultValue != null &&
                                    node.Attr[key.Key] == attr_def.DefaultValue)
                                node.Attr[key.Key].ClearValue();
                        }
                    }
                }
            }
        }

        private static void _RemoveDefaultAttrs(OpList producer_op_list, GraphDef graph_def)
        {
            var producer_op_dict = producer_op_list.Op.ToDictionary(x => x.Name, x => x);

            foreach (var node in graph_def.Node)
            {
                // Remove any default attr values that aren't in op_def.
                if (producer_op_dict.ContainsKey(node.Op))
                {
                    var op_def = op_def_registry.GetOpDef(node.Op);
                    if(op_def is null)
                    {
                        continue;
                    }
                    var producer_op_def = producer_op_dict[node.Op];
                    foreach (var key in node.Attr.Keys)
                    {
                        if (_FindAttrInOpDef(key, op_def) is null)
                        {
                            var attr_def = _FindAttrInOpDef(key, producer_op_def);
                            if (attr_def != null && attr_def.DefaultValue != null &&
                                    node.Attr[key] == attr_def.DefaultValue)
                                node.Attr[key].ClearValue();
                        }
                    }
                }
            }
        }

        private static AttrDef _FindAttrInOpDef(string name, OpDef op_def)
        {
            return op_def.Attr.FirstOrDefault(x => x.Name == name);
        }
    }
}
