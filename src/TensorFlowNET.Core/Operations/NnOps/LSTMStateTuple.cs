using System;
using System.Collections.Generic;
using System.Text;

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
    public class LSTMStateTuple
    {
        int c;
        int h;

        public LSTMStateTuple(int c)
        {
            this.c = c;
        }

        public LSTMStateTuple(int c, int h)
        {
            this.c = c;
            this.h = h;
        }

        public static implicit operator int(LSTMStateTuple tuple)
        {
            return tuple.c;
        }

        public static implicit operator LSTMStateTuple(int c)
        {
            return new LSTMStateTuple(c);
        }
    }
}
