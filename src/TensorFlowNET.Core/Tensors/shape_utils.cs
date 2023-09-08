using System;
using System.Linq;
using Tensorflow.Eager;
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

        public static Shape from_object_array(object[] shape)
        {
            var dims = shape.Select(x =>
            {
                if (x is KerasTensor kt && kt.inferred_value != null)
                {
                    return kt.inferred_value.as_int_list()[0];
                }
                else if (x is EagerTensor et && et.dtype == TF_DataType.TF_INT32)
                {
                    return et.ToArray<int>()[0];
                }
                else if (x is int i)
                {
                    return i;
                }
                else if (x is long l)
                {
                    return l;
                }
                throw new NotImplementedException();
            }).ToArray();

            return new Shape(dims);
        }
    }
}
