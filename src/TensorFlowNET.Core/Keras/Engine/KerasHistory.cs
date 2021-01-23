namespace Tensorflow.Keras.Engine
{
    /// <summary>
    /// Tracks the Layer call that created a Tensor, for Keras Graph Networks.
    /// </summary>
    public class KerasHistory
    {
        ILayer layer;
        public ILayer Layer => layer;
        int node_index;
        public int NodeIndex => node_index;
        int tensor_index;
        public int TensorIndex => tensor_index;

        public KerasHistory(ILayer layer, int node_index, int tensor_index)
        {
            this.layer = layer;
            this.node_index = node_index;
            this.tensor_index = tensor_index;
        }

        public void Deconstruct(out ILayer layer, out int node_index, out int tensor_index)
        {
            layer = this.layer;
            node_index = this.node_index;
            tensor_index = this.tensor_index;
        }

        public override string ToString()
            => $"{layer.GetType().Name} {layer.Name}";
    }
}
