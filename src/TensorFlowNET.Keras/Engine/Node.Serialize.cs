using System;
using System.Collections.Generic;
using Tensorflow.Keras.Saving;

namespace Tensorflow.Keras.Engine
{
    public partial class Node
    {
        /// <summary>
        /// Serializes `Node` for Functional API's `get_config`.
        /// </summary>
        /// <returns></returns>
        public NodeConfig serialize(Func<string, int, string> make_node_key, Dictionary<string, int> node_conversion_map)
        {
            throw new NotImplementedException("");
        }
    }
}
