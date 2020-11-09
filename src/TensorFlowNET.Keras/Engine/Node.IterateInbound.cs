using System.Collections.Generic;

namespace Tensorflow.Keras.Engine
{
    public partial class Node
    {
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
