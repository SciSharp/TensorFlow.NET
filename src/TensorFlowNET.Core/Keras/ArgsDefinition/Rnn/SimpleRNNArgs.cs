namespace Tensorflow.Keras.ArgsDefinition.Rnn
{
    public class SimpleRNNArgs : RNNArgs
    {
        public float Dropout = 0f;
        public float RecurrentDropout = 0f;
        public int state_size;
        public int output_size;
    }
}
