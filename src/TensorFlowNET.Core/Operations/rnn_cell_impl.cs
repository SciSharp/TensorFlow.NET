using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Operations
{
    public class rnn_cell_impl
    {
        public BasicRNNCell BasicRNNCell(int num_units)
            => new BasicRNNCell(num_units);
    }
}
