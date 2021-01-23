using System;
using System.Collections.Generic;
using Tensorflow.Keras.Saving;

namespace Tensorflow.Keras.Engine
{
    public interface INode
    {
        Tensors input_tensors { get; }
        Tensors Outputs { get; }
        ILayer Layer { get; }
        List<Tensor> KerasInputs { get; set; }
        INode[] ParentNodes { get; }
        IEnumerable<(ILayer, int, int, Tensor)> iterate_inbound();
        bool is_input { get; }
        List<NodeConfig> serialize(Func<string, int, string> make_node_key, Dictionary<string, int> node_conversion_map);
    }
}
