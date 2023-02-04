namespace Tensorflow.Keras.ArgsDefinition
{
    // TODO: no corresponding class found in keras python, maybe obselete?
    public class ResizingArgs : PreprocessingLayerArgs
    {
        public int Height { get; set; }
        public int Width { get; set; }
        public string Interpolation { get; set; } = "bilinear";
    }
}
