namespace Tensorflow.Keras.ArgsDefinition
{
    public class ReshapeArgs : LayerArgs
    {
        public TensorShape TargetShape { get; set; }
        public object[] TargetShapeObjects { get; set; }
    }
}
