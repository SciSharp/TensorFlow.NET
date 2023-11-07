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
        public static Functional from_config(FunctionalConfig config)
        {
            var (input_tensors, output_tensors, created_layers) = reconstruct_from_config(config);
            var model = new Functional(input_tensors, output_tensors, name: config.Name);
            model.connect_ancillary_layers(created_layers);
            return model;
        }

        /// <summary>
        /// Reconstructs graph from config object.
        /// </summary>
        /// <param name="config"></param>
        /// <returns></returns>
        public static (Tensors, Tensors, Dictionary<string, ILayer>) reconstruct_from_config(FunctionalConfig config, Dictionary<string, ILayer>? created_layers = null)
        {
            // Layer instances created during the graph reconstruction process.
            created_layers = created_layers ?? new Dictionary<string, ILayer>();
            var node_index_map = new Dictionary<(string, int), int>();
            var node_count_by_layer = new Dictionary<ILayer, int>();
            var unprocessed_nodes = new Dictionary<ILayer, List<NodeConfig>>();
            // First, we create all layers and enqueue nodes to be processed
            foreach (var layer_data in config.Layers)
                process_layer(created_layers, layer_data, unprocessed_nodes, node_count_by_layer);

            // Then we process nodes in order of layer depth.
            // Nodes that cannot yet be processed (if the inbound node
            // does not yet exist) are re-enqueued, and the process
            // is repeated until all nodes are processed.
            while (unprocessed_nodes.Count > 0)
            {
                foreach(var layer_data in config.Layers)
                {
                    var layer = created_layers[layer_data.Name];
                    if (unprocessed_nodes.ContainsKey(layer))
                    {
                        var node_data = unprocessed_nodes[layer];
                        // foreach (var node_data in unprocessed_nodes[layer])
                        {
                            process_node(layer, node_data, created_layers, node_count_by_layer, node_index_map);
                            unprocessed_nodes.Remove(layer);
                        }
                    }
                }
            }

            var input_tensors = new List<Tensor>();
            foreach (var layer_data in config.InputLayers)
            {
                var (layer_name, node_index, tensor_index) = (layer_data.Name, layer_data.NodeIndex, layer_data.TensorIndex);
                var layer = created_layers[layer_name];
                var layer_output_tensors = layer.InboundNodes[node_index].Outputs;
                input_tensors.append(layer_output_tensors[tensor_index]);
            }

            var output_tensors = new List<Tensor>();
            foreach (var layer_data in config.OutputLayers)
            {
                var (layer_name, node_index, tensor_index) = (layer_data.Name, layer_data.NodeIndex, layer_data.TensorIndex);
                var layer = created_layers[layer_name];
                var layer_output_tensors = layer.InboundNodes[node_index].Outputs;
                output_tensors.append(layer_output_tensors[tensor_index]);
            }

            return (input_tensors, output_tensors, created_layers);
        }

        static void process_layer(Dictionary<string, ILayer> created_layers, 
            LayerConfig layer_data, 
            Dictionary<ILayer, List<NodeConfig>> unprocessed_nodes,
            Dictionary<ILayer, int> node_count_by_layer)
        {
            ILayer layer = null;
            var layer_name = layer_data.Name;
            if (created_layers.ContainsKey(layer_name))
                layer = created_layers[layer_name];
            else
            {
                layer = generic_utils.deserialize_keras_object(layer_data.ClassName, layer_data.Config);

                created_layers[layer_name] = layer;
            }
            node_count_by_layer[layer] = layer_data.InboundNodes.Count - (_should_skip_first_node(layer) ? 1 : 0);

            var inbound_nodes_data = layer_data.InboundNodes;
            foreach (var node_data in inbound_nodes_data)
            {
                if (!unprocessed_nodes.ContainsKey(layer))
                    unprocessed_nodes[layer] = new List<NodeConfig>() { node_data };
                else
                    unprocessed_nodes[layer].Add(node_data);
            }
        }

        static void process_node(ILayer layer, 
            List<NodeConfig> nodes_data, 
            Dictionary<string, ILayer> created_layers,
            Dictionary<ILayer, int> node_count_by_layer,
            Dictionary<(string, int), int> node_index_map)
        {

            var input_tensors = new List<Tensor>();

            for (int i = 0; i < nodes_data.Count; i++)
            {
                var node_data = nodes_data[i];
                var inbound_layer_name = node_data.Name;
                var inbound_node_index = node_data.NodeIndex;
                var inbound_tensor_index = node_data.TensorIndex;

                var inbound_layer = created_layers[inbound_layer_name];
                var inbound_node = inbound_layer.InboundNodes[inbound_node_index];
                input_tensors.Add(inbound_node.Outputs[inbound_node_index]);
            }

            var output_tensors = layer.Apply(input_tensors);

            // Update node index map.
            var output_index = output_tensors[0].KerasHistory.NodeIndex;
            node_index_map[(layer.Name, node_count_by_layer[layer])] = output_index;
            node_count_by_layer[layer] += 1;
        }

        static bool _should_skip_first_node(ILayer layer)
        {
            return layer is Functional && layer.Layers[0] is InputLayer;
        }
    }
}
