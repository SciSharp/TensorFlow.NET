using System;
using System.Collections.Generic;
using System.Linq;
using Tensorflow.Keras.Saving;
using static Tensorflow.Binding;

namespace Tensorflow.Keras.Engine
{
    public partial class Node
    {
        /// <summary>
        /// Serializes `Node` for Functional API's `get_config`.
        /// </summary>
        /// <returns></returns>
        public List<NodeConfig> serialize(Func<string, int, string> make_node_key, Dictionary<string, int> node_conversion_map)
        {
            return KerasInputs.Select(x => {
                var kh = x.KerasHistory;
                var node_key = make_node_key(kh.Layer.Name, kh.NodeIndex);
                var new_node_index = node_conversion_map.Get(node_key, 0);
                return new NodeConfig
                {
                    Name = kh.Layer.Name,
                    NodeIndex = new_node_index,
                    TensorIndex = kh.TensorIndex
                };
            }).ToList();
        }
    }
}
