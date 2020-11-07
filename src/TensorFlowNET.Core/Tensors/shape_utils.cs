using System;
using System.Linq;
using static Tensorflow.Binding;

namespace Tensorflow
{
    public class shape_utils
    {
        public static Tensor static_or_dynamic_map_fn(Func<Tensor, Tensor> fn, Tensor elems, TF_DataType[] dtypes = null,
            int parallel_iterations = 32, bool back_prop = true)
        {
            var outputs = tf.unstack(elems).Select(arg => fn(arg)).ToArray();

            throw new NotImplementedException("");
        }
    }
}
