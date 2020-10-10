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
        public Tensor tensor;

        public KerasHistory(Layer layer, int node_index, int tensor_index, Tensor tensor)
        {
            this.layer = layer;
            this.node_index = node_index;
            this.tensor_index = tensor_index;
            this.tensor = tensor;
            Console.WriteLine(tensor.name);
        }

        public void Deconstruct(out Layer layer, out int node_index, out int tensor_index)
        {
            layer = this.layer;
            node_index = this.node_index;
            tensor_index = this.tensor_index;
        }

        public override string ToString()
            => $"{layer.GetType().Name} {layer.Name} {tensor.name}";

        public static implicit operator Layer(KerasHistory history)
            => history.layer;
    }
}
