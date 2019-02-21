using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow
{
    public class gen_training_ops
    {
        public static OpDefLibrary _op_def_lib = new OpDefLibrary();

        public static Tensor apply_gradient_descent(RefVariable var, Tensor alpha, Tensor delta, bool use_locking = false, string name = null)
        {
            var _op = _op_def_lib._apply_op_helper("ApplyGradientDescent", name, new
            {
                var,
                alpha,
                delta,
                use_locking
            });

            return _op.outputs[0];
        }
    }
}
