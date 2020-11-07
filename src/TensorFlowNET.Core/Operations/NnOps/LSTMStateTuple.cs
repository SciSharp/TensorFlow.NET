namespace Tensorflow.Operations
{
    /// <summary>
    /// Tuple used by LSTM Cells for `state_size`, `zero_state`, and output state.
    /// 
    /// Stores two elements: `(c, h)`, in that order. Where `c` is the hidden state
    /// and `h` is the output.
    /// 
    /// Only used when `state_is_tuple=True`.
    /// </summary>
    public class LSTMStateTuple : ICanBeFlattened
    {
        public object c;
        public object h;

        public LSTMStateTuple(int c, int h)
        {
            this.c = c;
            this.h = h;
        }

        public LSTMStateTuple(Tensor c, Tensor h)
        {
            this.c = c;
            this.h = h;
        }

        public object[] Flatten()
            => new[] { c, h };
    }
}
