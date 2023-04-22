using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Tensorflow.Keras.Layers;
using Tensorflow.Keras.Saving;
using Tensorflow.Keras.Utils;
using static Tensorflow.Binding;

namespace Tensorflow.Keras.Engine
{
    public partial class Functional
    {
        public override IKerasConfig get_config()
        {
            return get_network_config();
        }

        /// <summary>
        /// Builds the config, which consists of the node graph and serialized layers.
        /// </summary>
        FunctionalConfig get_network_config()
        {
            var config = new FunctionalConfig
            {
                Name = name
            };
            
            var node_conversion_map = new Dictionary<string, int>();
            foreach (var layer in _self_tracked_trackables)
            {
                var kept_nodes = _should_skip_first_node(layer) ? 1 : 0;
                foreach (var (original_node_index, node) in enumerate(layer.InboundNodes))
                {
                    var node_key = _make_node_key(layer.Name, original_node_index);
                    if (NetworkNodes.Contains(node_key))
                    {
                        node_conversion_map[node_key] = kept_nodes;
                        kept_nodes += 1;
                    }
                }
            }

            var layer_configs = new List<LayerConfig>();
            using (SharedObjectSavingScope.Enter())
            {
                foreach (var layer in _self_tracked_trackables)
                {
                    var filtered_inbound_nodes = new List<NodeConfig>();
                    foreach (var (original_node_index, node) in enumerate(layer.InboundNodes))
                    {
                        var node_key = _make_node_key(layer.Name, original_node_index);
                        if (NetworkNodes.Contains(node_key) && !node.is_input)
                        {
                            var node_data = node.serialize(_make_node_key, node_conversion_map);
                            filtered_inbound_nodes.append(node_data);
                        }
                    }

                    var layer_config = generic_utils.serialize_layer_to_config(layer);
                    layer_config.Name = layer.Name;
                    layer_config.InboundNodes = filtered_inbound_nodes;
                    layer_configs.Add(layer_config);
                }
            }
            config.Layers = layer_configs;

            // Gather info about inputs and outputs.
            var model_inputs = new List<NodeConfig>();
            foreach (var i in range(_input_layers.Count))
            {
                var (layer, node_index, tensor_index) = _input_coordinates[i];
                var node_key = _make_node_key(layer.Name, node_index);
                if (!NetworkNodes.Contains(node_key))
                    continue;
                var new_node_index = node_conversion_map[node_key];
                model_inputs.append(new NodeConfig
                {
                    Name = layer.Name,
                    NodeIndex = new_node_index,
                    TensorIndex = tensor_index
                });
            }
            config.InputLayers = model_inputs;

            var model_outputs = new List<NodeConfig>();
            foreach (var i in range(_output_layers.Count))
            {
                var (layer, node_index, tensor_index) = _output_coordinates[i];
                var node_key = _make_node_key(layer.Name, node_index);
                if (!NetworkNodes.Contains(node_key))
                    continue;
                var new_node_index = node_conversion_map[node_key];
                model_outputs.append(new NodeConfig
                {
                    Name = layer.Name,
                    NodeIndex = new_node_index,
                    TensorIndex = tensor_index
                });
            }
            config.OutputLayers = model_outputs;

            return config;
        }

        string _make_node_key(string layer_name, int node_index)
            => $"{layer_name}_ib-{node_index}";
    }
}
