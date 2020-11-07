namespace Tensorflow.Keras.ArgsDefinition
{
    public class ModelArgs : LayerArgs
    {
        public Tensors Inputs { get; set; }
        public Tensors Outputs { get; set; }
    }
}
