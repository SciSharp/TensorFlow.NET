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
    }
}
