namespace Tensorflow.Keras.ArgsDefinition
{
    public class ResizingArgs : PreprocessingLayerArgs
    {
        public int Height { get; set; }
        public int Width { get; set; }
        public string Interpolation { get; set; } = "bilinear";
    }
}
