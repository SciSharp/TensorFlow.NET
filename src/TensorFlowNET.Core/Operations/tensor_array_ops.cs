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
            if (!tf.Context.executing_eagerly() && old_ta is not _GraphTensorArrayV2 && control_flow_util.EnableControlFlowV2(ops.get_default_graph()))
            {
                throw new NotImplementedException("Attempting to build a graph-mode TF2-style "
                                + "TensorArray from either an eager-mode "
                                + "TensorArray or a TF1-style TensorArray.  "
                                + "This is not currently supported.  You may be "
                                + "attempting to capture a TensorArray "
                                + "inside a tf.function or tf.data map function. "
                                + "Instead, construct a new TensorArray inside "
                                + "the function.");
            }
            var new_ta = TensorArray.Create(old_ta.dtype, handle: old_ta.handle, flow: flow, infer_shape: old_ta.infer_shape,
                colocate_with_first_write_call: old_ta.colocate_with_first_write_call);
            new_ta._dynamic_size = old_ta._dynamic_size;
            new_ta._size = old_ta._size;
            new_ta._colocate_with = old_ta._colocate_with;
            new_ta._element_shape = old_ta._element_shape;
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
