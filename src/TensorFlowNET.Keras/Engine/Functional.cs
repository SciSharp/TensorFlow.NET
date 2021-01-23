using System;
using System.Collections.Generic;
using System.Linq;
using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Keras.Utils;
using static Tensorflow.Binding;

namespace Tensorflow.Keras.Engine
{
    /// <summary>
    /// A `Functional` model is a `Model` defined as a directed graph of layers.
    /// </summary>
    public partial class Functional : Model
    {
        TensorShape _build_input_shape;
        bool _compute_output_and_mask_jointly;
        bool _expects_training_arg;
        bool _expects_mask_arg;
        bool _autocast;
        List<ILayer> _output_layers;
        List<ILayer> _input_layers;
        List<KerasHistory> _input_coordinates;
        List<KerasHistory> _output_coordinates;
        public string[] NetworkNodes { get; set; }

        Dictionary<long, int> tensor_usage_count;

        public Functional(Tensors inputs, Tensors outputs, string name = null)
            : base(new ModelArgs
            {
                Name = name,
                Inputs = inputs,
                Outputs = outputs
            })
        {
            _input_layers = new List<ILayer>();
            _output_layers = new List<ILayer>();
            _input_coordinates = new List<KerasHistory>();
            _output_coordinates = new List<KerasHistory>();
            tensor_usage_count = new Dictionary<long, int>();
            if (this is Sequential)
                return;
            _init_graph_network(inputs, outputs);
        }

        protected void _init_graph_network(Tensors inputs, Tensors outputs)
        {
            _is_graph_network = true;
            this.inputs = inputs;
            this.outputs = outputs;
            built = true;
            _build_input_shape = inputs.shape;
            _compute_output_and_mask_jointly = true;
            _expects_training_arg = true;
            _expects_mask_arg = true;
            // A graph network does not autocast inputs, as its layers will cast them instead.
            _autocast = false;

            if (outputs.Any(x => x.KerasHistory == null))
                base_layer_utils.create_keras_history(outputs);

            // Build self._output_layers:
            foreach (var x in outputs)
            {
                var (layer, node_index, tensor_index) = x.KerasHistory;
                _output_layers.append(layer);
                _output_coordinates.append(new KerasHistory(layer, node_index, tensor_index));
            }

            // Build self._input_layers:
            foreach (var x in inputs)
            {
                var (layer, node_index, tensor_index) = x.KerasHistory;
                _input_layers.append(layer);
                _input_coordinates.append(new KerasHistory(layer, node_index, tensor_index));
            }

            // Keep track of the network's nodes and layers.
            var (nodes, nodes_by_depth, layers, _) = MapGraphNetwork(inputs, outputs);

            NetworkNodes = nodes;
            NodesByDepth = nodes_by_depth;
            _layers = layers;

            // Build self.input_names and self.output_names.
            _set_output_names();

            ComputeTensorUsageCount();
        }

        /// <summary>
        /// Assigns unique names to the Network's outputs.
        /// </summary>
        void _set_output_names()
        {
            var uniquified = new List<string>();
            var output_names = new List<string>();
            var prefix_count = new Dictionary<string, int>();

            foreach (var layer in _output_layers)
            {
                var proposal = layer.Name;
                while (output_names.Contains(proposal))
                {
                    var existing_count = prefix_count.Get(layer.Name, 1);
                    proposal = $"{layer.Name}_{existing_count}";
                    prefix_count[layer.Name] = existing_count + 1;
                }
                output_names.add(proposal);
                uniquified.append(proposal);
            }

            this.output_names = uniquified.ToArray();
        }

        void ComputeTensorUsageCount()
        {
            var available_tensors = inputs.Select(x => x.Id).ToList();
            var depth_keys = NodesByDepth.Keys.OrderBy(x => x).Reverse().Skip(1).ToArray();
            foreach (var depth in depth_keys)
            {
                foreach (var node in NodesByDepth[depth])
                {
                    var input_tensors = node.KerasInputs.Select(x => x.Id).ToArray();
                    if (input_tensors.issubset(available_tensors))
                    {
                        foreach (var tensor in node.KerasInputs)
                        {
                            if (!tensor_usage_count.ContainsKey(tensor.Id))
                                tensor_usage_count[tensor.Id] = 0;
                            tensor_usage_count[tensor.Id] += 1;
                        }

                        foreach (var output_tensor in node.Outputs)
                            available_tensors.Add(output_tensor.Id);
                    }
                }
            }

            foreach (var tensor in outputs)
            {
                if (!tensor_usage_count.ContainsKey(tensor.Id))
                    tensor_usage_count[tensor.Id] = 0;
                tensor_usage_count[tensor.Id] += 1;
            }
        }

        /// <summary>
        /// Validates a network's topology and gather its layers and nodes.
        /// </summary>
        /// <param name="inputs"></param>
        /// <param name="outputs"></param>
        (string[], Dictionary<int, List<INode>>, List<ILayer>, Dictionary<int, List<ILayer>>) MapGraphNetwork(Tensors inputs, Tensors outputs)
        {
            var (nodes_in_decreasing_depth, layer_indices) = BuildMap(outputs);
            var network_nodes = nodes_in_decreasing_depth
                .Select(node => MakeNodeKey(node.Layer.Name, node.Layer.InboundNodes.IndexOf(node)))
                .ToArray();

            var nodes_depths = new Dictionary<INode, int>();
            var layers_depths = new Dictionary<ILayer, int>();

            nodes_in_decreasing_depth.Reverse();
            foreach (var node in nodes_in_decreasing_depth)
            {
                // If the depth is not set, the node has no outbound nodes (depth 0).
                int depth = nodes_depths.SetDefault(node, 0);
                // Update the depth of the corresponding layer
                int previous_depth = layers_depths.Get(node.Layer, 0);
                // If we've seen this layer before at a higher depth,
                // we should use that depth instead of the node depth.
                // This is necessary for shared layers that have inputs at different
                // depth levels in the graph.
                depth = Math.Max(depth, previous_depth);
                layers_depths[node.Layer] = depth;
                nodes_depths[node] = depth;

                // Update the depth of inbound nodes.
                // The "depth" of a node is the max of the depths
                // of all nodes it is connected to + 1.
                foreach (var node_dep in node.ParentNodes)
                {
                    previous_depth = nodes_depths.Get(node_dep, 0);
                    nodes_depths[node_dep] = Math.Max(depth + 1, previous_depth);
                }
            }

            // Handle inputs that are not connected to outputs.
            // We do not error out here because the inputs may be used to compute losses
            // and metrics.
            foreach (var input_t in inputs)
            {
                var (input_layer, _, _) = input_t.KerasHistory;
                if (!layers_depths.ContainsKey(input_layer))
                {
                    layers_depths[input_layer] = 0;
                    layer_indices[input_layer] = -1;
                    nodes_depths[input_layer.InboundNodes[0]] = 0;
                    network_nodes.add(MakeNodeKey(input_layer.Name, 0));
                }
            }

            // Build a dict {depth: list of nodes with this depth}
            var nodes_by_depth = new Dictionary<int, List<INode>>();
            foreach (var (node, depth) in enumerate(nodes_depths))
            {
                if (!nodes_by_depth.ContainsKey(depth))
                    nodes_by_depth[depth] = new List<INode>();
                nodes_by_depth[depth].append(node);
            }

            var layers_by_depth = new Dictionary<int, List<ILayer>>();
            foreach (var (layer, depth) in enumerate(layers_depths))
            {
                if (!layers_by_depth.ContainsKey(depth))
                    layers_by_depth[depth] = new List<ILayer>();
                layers_by_depth[depth].append(layer);
            }

            // Get sorted list of layer depths.
            var depth_keys = layers_by_depth.Keys.OrderBy(x => x).Reverse();

            // Set self.layers ordered by depth.
            var layers = new List<ILayer>();
            foreach (var depth in depth_keys)
            {
                var layers_for_depth = layers_by_depth[depth];

                // Network.layers needs to have a deterministic order:
                // here we order them by traversal order.
                layers_for_depth = layers_for_depth.OrderBy(x => layer_indices[x]).ToList();
                layers.AddRange(layers_for_depth);
            }

            // Get sorted list of node depths.
            depth_keys = nodes_by_depth.Keys.OrderBy(x => x).Reverse();

            return (network_nodes, nodes_by_depth, layers, layers_by_depth);
        }

        string MakeNodeKey(string layer_name, int node_index)
            => $"{layer_name}_ib-{node_index}";

        /// <summary>
        /// This method topologically sorts nodes in order from inputs to outputs.
        /// </summary>
        /// <param name="outputs"></param>
        (List<INode>, Dictionary<ILayer, int>) BuildMap(Tensors outputs)
        {
            var finished_nodes = new List<INode>();
            var nodes_in_progress = new List<INode>();
            var nodes_in_decreasing_depth = new List<INode>();
            var layer_indices = new Dictionary<ILayer, int>();
            foreach (var output in outputs)
                BuildMapHelper(output,
                    finished_nodes,
                    nodes_in_progress,
                    nodes_in_decreasing_depth,
                    layer_indices);

            return (nodes_in_decreasing_depth, layer_indices);
        }

        void BuildMapHelper(Tensor tensor,
            List<INode> finished_nodes,
            List<INode> nodes_in_progress,
            List<INode> nodes_in_decreasing_depth,
            Dictionary<ILayer, int> layer_indices)
        {
            var (layer, node_index, _) = tensor.KerasHistory;
            var node = layer.InboundNodes[node_index] as Node;

            // Don't repeat work for shared subgraphs
            if (finished_nodes.Contains(node))
                return;

            // Prevent cycles.
            if (nodes_in_progress.Contains(node))
                throw new ValueError($"The tensor {tensor.name} at layer {layer.Name} is part of a cycle.");

            // Store the traversal order for layer sorting.
            if (!layer_indices.ContainsKey(layer))
                layer_indices[layer] = layer_indices.Count;

            // Propagate to all previous tensors connected to this node.
            nodes_in_progress.Add(node);
            if (!node.is_input)
            {
                foreach (var k_tensor in node.KerasInputs)
                {
                    BuildMapHelper(k_tensor,
                        finished_nodes,
                        nodes_in_progress,
                        nodes_in_decreasing_depth,
                        layer_indices);
                }
            }

            finished_nodes.Add(node);
            nodes_in_progress.Remove(node);
            nodes_in_decreasing_depth.append(node);
        }

        protected override Tensors Call(Tensors inputs, Tensor state = null, bool is_training = false)
        {
            return run_internal_graph(inputs, is_training);
        }

        Tensors run_internal_graph(Tensors inputs, bool training = false, Tensors mask = null)
        {
            if (mask == null)
            {
                Tensor[] masks = new Tensor[inputs.Count()];
                foreach (var (i, input_t) in enumerate(inputs))
                    input_t.KerasMask = masks[i];
            }

            var tensor_dict = new Dictionary<long, Queue<Tensor>>();
            foreach (var (x, y) in zip(this.inputs, inputs))
            {
                var y1 = conform_to_reference_input(y, x);
                tensor_dict[x.Id] = new Queue<Tensor>(Enumerable.Range(0, tensor_usage_count[x.Id]).Select(x => y1));
            }

            var depth_keys = NodesByDepth.Keys.OrderBy(x => x).Reverse().ToArray();

            foreach (var depth in depth_keys)
            {
                var nodes = NodesByDepth[depth];
                foreach (Node node in nodes)
                {
                    // Input tensors already exist.
                    if (node.is_input)
                        continue;

                    var layer_inputs = node.MapArguments(tensor_dict);

                    tf.Logger.Debug($"Depth {depth}: {node.Layer}: {node.Layer.Name}");
                    var outputs = node.Layer.Apply(layer_inputs, is_training: training);
                    foreach (var output in outputs.Where(x => x != null))
                        tf.Logger.Information($"Depth {depth}: {node.Layer}: {node.Layer.Name} {output.TensorShape}");
                    // Update tensor_dict for next input
                    foreach (var (x_id, y) in zip(node.FlatOutputIds, outputs))
                        tensor_dict[x_id] = new Queue<Tensor>(Enumerable.Range(0, tensor_usage_count[x_id]).Select(x => y));
                }
            }

            var output_tensors = new Tensors();

            foreach (var x in outputs)
                output_tensors.Add(tensor_dict[x.Id].Dequeue());

            return output_tensors;
        }

        Tensor conform_to_reference_input(Tensor tensor, Tensor ref_input)
        {
            return tensor;
        }
    }
}
