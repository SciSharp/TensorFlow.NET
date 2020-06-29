﻿/*****************************************************************************
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
using static Tensorflow.OpDef.Types;
using static Tensorflow.Binding;

namespace Tensorflow
{
    public class importer
    {
        public static ITensorOrOperation[] import_graph_def(GraphDef graph_def,
            Dictionary<string, Tensor> input_map = null,
            string[] return_elements = null,
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
            using (var buffer = c_api_util.tf_buffer(bytes))
            using (var scoped_options = c_api_util.ScopedTFImportGraphDefOptions())
            using (var status = new Status())
            {
                _PopulateTFImportGraphDefOptions(scoped_options, prefix, input_map, return_elements);
                // need to create a class ImportGraphDefWithResults with IDisposal
                results = c_api.TF_GraphImportGraphDefWithResults(graph, buffer.Handle, scoped_options.Handle, status.Handle);
                status.Check(true);
            }

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
            int opers_idx = 0;
            foreach(var name in requested_return_elements)
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
            foreach(var new_op in graph._add_new_tf_operations())
            {
                var original_device = new_op.Device;
            }
        }

        public static void _PopulateTFImportGraphDefOptions(ImportGraphDefOptions options, 
            string prefix, 
            Dictionary<string, Tensor> input_map,
            string[] return_elements)
        {
            c_api.TF_ImportGraphDefOptionsSetPrefix(options.Handle, prefix);
            c_api.TF_ImportGraphDefOptionsSetUniquifyNames(options.Handle, (char)1);

            foreach(var input in input_map)
            {
                throw new NotImplementedException("_PopulateTFImportGraphDefOptions");
            }

            if (return_elements == null)
                return_elements = new string[0];

            foreach (var name in return_elements)
            {
                if(name.Contains(":"))
                {
                    var (op_name, index) = _ParseTensorName(name);
                    c_api.TF_ImportGraphDefOptionsAddReturnOutput(options.Handle, op_name, index);
                }
                else
                {
                    c_api.TF_ImportGraphDefOptionsAddReturnOperation(options.Handle, name);
                }
            }

            // c_api.TF_ImportGraphDefOptionsSetValidateColocationConstraints(options, validate_colocation_constraints);
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
            foreach(var node in graph_def.Node)
            {
                if (!op_dict.ContainsKey(node.Op))
                    continue;

                var op_def = op_dict[node.Op];
                _SetDefaultAttrValues(node, op_def);
            }

            return graph_def;
        }

        private static void _SetDefaultAttrValues(NodeDef node_def, OpDef op_def)
        {
            foreach(var attr_def in op_def.Attr)
            {
                var key = attr_def.Name;
                if(attr_def.DefaultValue != null)
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

            foreach(var node in graph_def.Node)
            {
                // Remove any default attr values that aren't in op_def.
                if (producer_op_dict.ContainsKey(node.Op))
                {
                    var op_def = op_dict[node.Op];
                    var producer_op_def = producer_op_dict[node.Op];
                    foreach(var key in node.Attr)
                    {
                        if(_FindAttrInOpDef(key.Key, op_def) == null)
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

        private static AttrDef _FindAttrInOpDef(string name, OpDef op_def)
        {
            return op_def.Attr.FirstOrDefault(x => x.Name == name);
        }
    }
}
