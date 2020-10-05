using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Keras.Engine
{
    /// <summary>
    /// Tracks the Layer call that created a Tensor, for Keras Graph Networks.
    /// </summary>
    public class KerasHistory
    {
        Layer layer;
        int node_index;
        int tensor_index;

        public KerasHistory(Layer layer, int node_index, int tensor_index)
        {
            this.layer = layer;
            this.node_index = node_index;
            this.tensor_index = tensor_index;
        }

        public static implicit operator Layer(KerasHistory history)
            => history.layer;

        public static implicit operator (Layer, int, int)(KerasHistory history)
            => (history.layer, history.node_index, history.tensor_index);
    }
}
