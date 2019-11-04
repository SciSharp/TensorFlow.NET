using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Operations;

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
            var impl = old_ta._implementation;

            var new_ta = new TensorArray(
                dtype: impl.dtype,
                handle: impl.handle,
                flow: flow,
                infer_shape: impl.infer_shape,
                colocate_with_first_write_call: impl.colocate_with_first_write_call);

            var new_impl = new_ta._implementation;
            new_impl._dynamic_size = impl._dynamic_size;
            new_impl._colocate_with = impl._colocate_with;
            new_impl._element_shape = impl._element_shape;
            return new_ta;
        }

        public static TensorArray build_ta_with_new_flow(_GraphTensorArray old_ta, Tensor flow)
        {
            var impl = old_ta;

            var new_ta = new TensorArray(
                dtype: impl.dtype,
                handle: impl.handle,
                flow: flow,
                infer_shape: impl.infer_shape,
                colocate_with_first_write_call: impl.colocate_with_first_write_call);

            var new_impl = new_ta._implementation;
            new_impl._dynamic_size = impl._dynamic_size;
            new_impl._colocate_with = impl._colocate_with;
            new_impl._element_shape = impl._element_shape;
            return new_ta;
        }
    }
}
