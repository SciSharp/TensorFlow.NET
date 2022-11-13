using Tensorflow.Keras.ArgsDefinition.Rnn;

namespace Tensorflow.Keras.ArgsDefinition.Lstm
{
    public class LSTMArgs : RNNArgs
    {
        public bool UnitForgetBias { get; set; }
        public float Dropout { get; set; }
        public float RecurrentDropout { get; set; }
        public int Implementation { get; set; }
    }
}
