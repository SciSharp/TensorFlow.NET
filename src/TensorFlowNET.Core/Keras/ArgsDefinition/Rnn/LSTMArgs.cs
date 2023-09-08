namespace Tensorflow.Keras.ArgsDefinition
{
    public class LSTMArgs : RNNArgs
    {
        // TODO: maybe change the `RNNArgs` and implement this class.
        public bool UnitForgetBias { get; set; }
        public int Implementation { get; set; }

        public LSTMArgs Clone()
        {
            return (LSTMArgs)MemberwiseClone();
        }
    }
}
