using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.RegularExpressions;
using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Keras.Engine;
using ThirdParty.Tensorflow.Python.Keras.Protobuf;
using static Tensorflow.Binding;

namespace Tensorflow.Keras.Saving
{
    public class KerasObjectLoader
    {
        SavedMetadata _metadata;
        SavedObjectGraph _proto;
        Dictionary<int, string> _node_paths = new Dictionary<int, string>();
        Dictionary<int, (Model, int[])> model_layer_dependencies = new Dictionary<int, (Model, int[])>();
        List<int> _traversed_nodes_from_config = new List<int>();

        public KerasObjectLoader(SavedMetadata metadata, SavedObjectGraph object_graph_def)
        {
            _metadata = metadata;
            _proto = object_graph_def;
            _metadata.Nodes.ToList().ForEach(x => _node_paths[x.NodeId] = x.NodePath);
        }

        /// <summary>
        /// Load all layer nodes from the metadata.
        /// </summary>
        /// <param name="compile"></param>
        public void load_layers(bool compile = true)
        {
            var metric_list = new List<ThirdParty.Tensorflow.Python.Keras.Protobuf.SavedObject>();
            foreach (var node_metadata in _metadata.Nodes)
            {
                if (node_metadata.Identifier == "_tf_keras_metric")
                {
                    metric_list.Add(node_metadata);
                    continue;
                }

                _load_layer(node_metadata.NodeId, node_metadata.Identifier, node_metadata.Metadata);
            }
        }

        void _load_layer(int node_id, string identifier, string metadata_json)
        {
            metadata_json = metadata_json.Replace("\"dtype\": \"float32\"", "\"dtype\": 1");
            var metadata = JsonConvert.DeserializeObject<KerasMetaData>(metadata_json);
            _revive_from_config(identifier, metadata, node_id);
        }

        /// <summary>
        /// Revives a layer/model from config, or returns None.
        /// </summary>
        /// <param name="identifier"></param>
        /// <param name="metadata"></param>
        /// <param name="node_id"></param>
        void _revive_from_config(string identifier, KerasMetaData metadata, int node_id)
        {
            var obj = _revive_graph_network(identifier, metadata, node_id);
            obj = obj ?? _revive_layer_or_model_from_config(metadata, node_id);
            _add_children_recreated_from_config(obj, _proto.Nodes[node_id], node_id);
        }

        Model _revive_graph_network(string identifier, KerasMetaData metadata, int node_id)
        {
            var config = metadata.Config;
            var class_name = metadata.ClassName;
            Model model = null;
            if (class_name == "Sequential")
            {
                model = new Sequential(new SequentialArgs
                {
                    Name = config.Name
                });
            }
            else if (class_name == "Functional")
            {
                throw new NotImplementedException("");
            }

            if (!metadata.IsGraphNetwork)
                return null;

            // Record this model and its layers. This will later be used to reconstruct
            // the model.
            var layers = _get_child_layer_node_ids(node_id);
            model_layer_dependencies[node_id] = (model, layers);
            return model;
        }

        Model _revive_layer_or_model_from_config(KerasMetaData metadata, int node_id)
        {
            var config = metadata.Config;
            var class_name = metadata.ClassName;
            var shared_object_id = metadata.SharedObjectId;
            var must_restore_from_config = metadata.MustRestoreFromConfig;

            return null;
        }

        /// <summary>
        /// Returns the node ids of each layer in a Sequential/Functional model.
        /// </summary>
        /// <param name="node_id"></param>
        int[] _get_child_layer_node_ids(int node_id)
        {
            int num_layers = 0;
            Dictionary<int, int> child_layers = new Dictionary<int, int>();
            foreach (var child in _proto.Nodes[node_id].Children)
            {
                var m = Regex.Match(child.LocalName, @"layer-(\d+)");
                if (!m.Success)
                    continue;
                var layer_n = int.Parse(m.Groups[1].Value);
                num_layers = max(layer_n + 1, num_layers);
                child_layers[layer_n] = child.NodeId;
            }

            var ordered = new List<int>();
            foreach (var n in range(num_layers))
            {
                if (child_layers.ContainsKey(n))
                    ordered.Add(child_layers[n]);
                else
                    break;
            }
            return ordered.ToArray();
        }

        /// <summary>
        /// Recursively records objects recreated from config.
        /// </summary>
        /// <param name="obj"></param>
        /// <param name="proto"></param>
        /// <param name="node_id"></param>
        void _add_children_recreated_from_config(Model obj, SavedObject proto, int node_id)
        {
            if (_traversed_nodes_from_config.Contains(node_id))
                return;
            var parent_path = _node_paths[node_id];
            _traversed_nodes_from_config.Add(node_id);
            if (!obj.Built)
            {
                var metadata_json = proto.UserObject.Metadata.Replace("\"dtype\": \"float32\"", "\"dtype\": 1");
                var metadata = JsonConvert.DeserializeObject<KerasMetaData>(metadata_json);
                _try_build_layer(obj, node_id, metadata.BuildInputShape);
            }
        }

        bool _try_build_layer(Model obj, int node_id, TensorShape build_input_shape)
        {
            if (obj.Built)
                return true;

            return false;
        }
    }
}
