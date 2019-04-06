using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow
{
    public class state_ops
    {
        /// <summary>
        /// Create a variable Operation.
        /// </summary>
        /// <param name="shape"></param>
        /// <param name="dtype"></param>
        /// <param name="name"></param>
        /// <param name="container"></param>
        /// <param name="shared_name"></param>
        /// <returns></returns>
        public static Tensor variable_op_v2(long[] shape,
            TF_DataType dtype,
            string name = "Variable",
            string container = "",
            string shared_name = "") => gen_state_ops.variable_v2(shape,
                dtype,
                name: name,
                container: container,
                shared_name: shared_name);

        public static Tensor assign(Tensor @ref, object value,
            bool validate_shape = true,
            bool use_locking = true,
            string name = null)
        {
            if (@ref.dtype.is_ref_dtype())
                return gen_state_ops.assign(@ref,
                    value,
                    validate_shape: validate_shape,
                    use_locking: use_locking,
                    name: name);
            else
                throw new NotImplementedException("state_ops.assign");
        }

        public static Tensor assign_sub(RefVariable @ref,
            Tensor value,
            bool use_locking = false,
            string name = null) => gen_state_ops.assign_sub(@ref,
                value,
                use_locking: use_locking,
                name: name);
    }
}
