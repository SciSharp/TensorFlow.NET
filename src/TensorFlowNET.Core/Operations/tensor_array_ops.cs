using Tensorflow.Operations;
using static Tensorflow.Binding;

namespace Tensorflow
{
    public class tensor_array_ops
    {
        /// <summary>
        /// Builds a TensorArray with a new `flow` tensor.
        /// </summary>
        /// <param name="old_ta"></param>
        /// <param name="flow"></param>
        /// <returns></returns>
        public static TensorArray build_ta_with_new_flow(TensorArray old_ta, Tensor flow)
        {
            var new_ta = tf.TensorArray(
                dtype: old_ta.dtype,
                infer_shape: old_ta.infer_shape,
                colocate_with_first_write_call: old_ta.colocate_with_first_write_call);

            return new_ta;
        }

        public static TensorArray build_ta_with_new_flow(_GraphTensorArray old_ta, Tensor flow)
        {
            var new_ta = tf.TensorArray(
                dtype: old_ta.dtype,
                infer_shape: old_ta.infer_shape,
                colocate_with_first_write_call: old_ta.colocate_with_first_write_call);

            return new_ta;
        }
    }
}
