namespace Tensorflow.Keras.ArgsDefinition
{
    public class ReshapeArgs : LayerArgs
    {
        public Shape TargetShape { get; set; }
        public object[] TargetShapeObjects { get; set; }
    }
}
