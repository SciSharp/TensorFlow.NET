using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Framework
{
    public class smart_module
    {
        public static object smart_cond(Tensor pred, 
            Func<(Tensor, Tensor, Tensor)> true_fn = null, 
            Func<(Tensor, Tensor, Tensor)> false_fn = null, 
            string name = null)
        {
            return control_flow_ops.cond(pred,
                true_fn: true_fn,
                false_fn: false_fn,
                name: name);
        }

        public static bool smart_constant_value(Tensor pred)
        {
            var pred_value = tensor_util.constant_value(pred);
            return pred_value;
        }
    }
}
