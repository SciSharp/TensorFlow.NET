using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Tensorflow.Keras.Utils
{
    public class tf_utils
    {
        public static bool are_all_symbolic_tensors(Tensor[] tensors)
        {
            return tensors.Select(x => is_symbolic_tensor(x)).Count() == tensors.Length;
        }

        public static bool is_symbolic_tensor(Tensor tensor)
        {
            return true;
        }
    }
}
