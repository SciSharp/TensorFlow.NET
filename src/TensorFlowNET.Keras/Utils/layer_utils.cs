using NumSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using Tensorflow.Keras.Engine;
using static Tensorflow.Binding;

namespace Tensorflow.Keras.Utils
{
    internal class layer_utils
    {
        public static void print_summary(Model model, int line_length = -1, float[] positions = null)
        {
            bool sequential_like = model is Sequential;
            //     || model.IsGraphNetwork;

            if (!sequential_like)
            {
                sequential_like = true;
                var nodes = new List<INode>();

                foreach (var v in model.NodesByDepth)
                {
                    // if the model has multiple nodes
                    // or if the nodes have multiple inbound_layers
                    // the model is no longer sequential
                    if (v.Value.Count > 1 || (v.Value.Count == 1 && v.Value[0].KerasInputs.Count > 1))
                    {
                        sequential_like = false;
                        break;
                    }

                    nodes.AddRange(v.Value);
                }

                if (sequential_like)
                {
                    // search for shared layers
                    foreach (var layer in model.Layers)
                    {
                        var flag = false;
                        foreach (var node in layer.InboundNodes)
                        {
                            if (nodes.Contains(node))
                            {
                                if (flag)
                                {
                                    sequential_like = false;
                                    break;
                                }
                                else
                                    flag = true;
                            }
                        }
                        if (!sequential_like)
                            break;
                    }
                }
            }

            string[] to_display;
            var relevant_nodes = new List<INode>();

            if (sequential_like)
            {
                if (line_length < 0)
                    line_length = 65;
                if (positions == null)
                    positions = new[] { 0.45f, 0.85f, 1.0f };
                if (positions.Last() <= 1)
                    positions = positions.Select(p => line_length * p).ToArray();
                to_display = new[] { "Layer (type)", "Output Shape", "Param #" };
            }
            else
            {
                if (line_length < 0)
                    line_length = 98;
                if (positions == null)
                    positions = new[] { 0.33f, 0.55f, 0.67f, 1.0f };
                if (positions.Last() <= 1)
                    positions = positions.Select(p => line_length * p).ToArray();
                to_display = new[] { "Layer (type)", "Output Shape", "Param #", "Connected to" };

                foreach (var v in model.NodesByDepth)
                    relevant_nodes.AddRange(v.Value);
            }

            int[] positions_int = positions.Select(x => Convert.ToInt32(x)).ToArray();
            print($"Model: {model.Name}");
            print(string.Join("", range(line_length).Select(x => "_")));
            print_row(to_display, positions_int);
            print(string.Join("", range(line_length).Select(x => "=")));

            foreach (var (i, layer) in enumerate(model.Layers))
            {
                if (sequential_like)
                    print_layer_summary(layer, positions_int);
                else
                    print_layer_summary_with_connections(layer, positions_int, relevant_nodes);
                if (i == model.Layers.Count - 1)
                    print(string.Join("", range(line_length).Select(x => "=")));
                else
                    print(string.Join("", range(line_length).Select(x => "_")));
            }

            var trainable_count = count_params(model, model.trainable_variables);
            var non_trainable_count = count_params(model, model.non_trainable_variables);

            print($"Total params: {trainable_count + non_trainable_count}");
            print($"Trainable params: {trainable_count}");
            print($"Non-trainable params: {non_trainable_count}");
            print(string.Join("", range(line_length).Select(x => "_")));
        }

        static void print_row(string[] fields, int[] positions)
        {
            var line = "";
            foreach (var i in range(fields.Length))
            {
                if (i > 0)
                    line = line + " ";
                line += fields[i];
                line = string.Join("", line.Take(positions[i]));
                line += string.Join("", range(positions[i] - len(line)).Select(x => " "));
            }
            print(line);
        }

        /// <summary>
        /// Prints a summary for a single layer.
        /// </summary>
        /// <param name="layer"></param>
        static void print_layer_summary(ILayer layer, int[] positions)
        {
            var name = layer.Name;

            var fields = new string[]
            {
                $"{name} ({layer.GetType().Name})",
                $"{layer.output_shape}",
                $"{layer.count_params()}"
            };

            print_row(fields, positions);
        }

        static void print_layer_summary_with_connections(ILayer layer, int[] positions, List<INode> relevant_nodes)
        {
            var connections = new List<string>();
            foreach (var node in layer.InboundNodes)
            {
                if (!relevant_nodes.Contains(node))
                    continue;

                foreach (var (inbound_layer, node_index, tensor_index, _) in node.iterate_inbound())
                    connections.append($"{inbound_layer.Name}[{node_index}][{tensor_index}]");
            }

            var name = layer.Name;
            string first_connection = "";
            if (connections.Count > 0)
                first_connection = connections[0];

            var fields = new string[]
            {
                $"{name}({layer.GetType().Name})",
                $"{layer.output_shape}",
                $"{layer.count_params()}",
                first_connection
            };

            print_row(fields, positions);

            if (connections.Count > 1)
            {
                foreach (var i in range(1, connections.Count))
                {
                    fields = new string[] { "", "", "", connections[i] };
                    print_row(fields, positions);
                }
            }
        }

        public static int count_params(Layer layer, List<IVariableV1> weights)
        {
            var weight_shapes = weights.Select(x => x.shape).ToArray();
            var total = weight_shapes.Select(p => (int)np.prod(p.dims)).Sum();
            return total;
        }

        public static Tensors get_source_inputs(Tensor tensor, ILayer layer = null,  int node_index = -1)
        {
            if (layer == null)
                (layer, node_index, _) = tensor.KerasHistory;
            if (layer.InboundNodes == null || layer.InboundNodes.Count == 0)
                return tensor;
            else
            {
                var node = layer.InboundNodes[node_index];
                if (node.is_input)
                    return node.input_tensors;
                else
                {
                    var source_tensors = new List<Tensor>();
                    foreach (var _layer in node.iterate_inbound())
                    {
                        (layer, node_index, tensor) = (_layer.Item1, _layer.Item2, _layer.Item4);
                        var previous_sources = get_source_inputs(tensor, layer, node_index);
                        foreach(var x in previous_sources)
                        {
                            // should be check if exist?
                            source_tensors.append(x);
                        }
                    }
                    return source_tensors;
                }
            }
        }
    }
}
