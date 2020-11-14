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
        public ModelConfig get_config()
        {
            return get_network_config();
        }

        /// <summary>
        /// Builds the config, which consists of the node graph and serialized layers.
        /// </summary>
        ModelConfig get_network_config()
        {
            var config = new ModelConfig
            {
                Name = name
            };

            var node_conversion_map = new Dictionary<string, int>();
            foreach (var layer in _layers)
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
            foreach (var layer in _layers)
            {
                var filtered_inbound_nodes = new List<INode>();
                foreach (var (original_node_index, node) in enumerate(layer.InboundNodes))
                {
                    var node_key = _make_node_key(layer.Name, original_node_index);
                    if (NetworkNodes.Contains(node_key) && !node.is_input)
                    {
                        var node_data = node.serialize(_make_node_key, node_conversion_map);
                        throw new NotImplementedException("");
                    }
                }

                var layer_config = generic_utils.serialize_keras_object(layer);
                layer_config.Name = layer.Name;
                layer_config.InboundNodes = filtered_inbound_nodes;
                layer_configs.Add(layer_config);
            }
            config.Layers = layer_configs;

            return config;
        }

        bool _should_skip_first_node(ILayer layer)
        {
            return layer is Functional && layer.Layers[0] is InputLayer;
        }

        string _make_node_key(string layer_name, int node_index)
            => $"{layer_name}_ib-{node_index}";
    }
}
