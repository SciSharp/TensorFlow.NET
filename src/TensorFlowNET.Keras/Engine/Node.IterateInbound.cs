using System.Collections.Generic;
using System.Linq;

namespace Tensorflow.Keras.Engine
{
    public partial class Node
    {
        public ILayer[] InboundLayers
            => iterate_inbound().Select(x => x.Item1).ToArray();

        public IEnumerable<(ILayer, int, int, Tensor)> iterate_inbound()
        {
            foreach (var kt in KerasInputs)
            {
                var (layer, node_index, tensor_index) = kt.KerasHistory;
                yield return (layer, node_index, tensor_index, kt);
            }
        }
    }
}
