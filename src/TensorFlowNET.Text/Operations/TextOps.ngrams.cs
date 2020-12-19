using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Text
{
    public partial class TextOps
    {
        public static Tensor ngrams(Tensor input, int width, 
            int axis = -1, 
            Reduction reduction_type = Reduction.None,
            string string_separator = " ",
            string name = null)
            => throw new NotImplementedException("");
    }
}
