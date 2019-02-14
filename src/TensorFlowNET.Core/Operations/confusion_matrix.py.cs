using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow
{
    public class confusion_matrix
    {
        public static (Tensor, Tensor, float) remove_squeezable_dimensions(Tensor labels,
            Tensor predictions,
            int expected_rank_diff = 0,
            string name = "")
        {
            throw new NotImplementedException("remove_squeezable_dimensions");
        }
    }
}
