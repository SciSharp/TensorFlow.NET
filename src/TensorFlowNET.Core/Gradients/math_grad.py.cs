using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow
{
    /// <summary>
    /// Gradients for operators defined in math_ops.py.
    /// </summary>
    public class math_grad
    {
        public static (Tensor, Tensor) _AddGrad(Operation op, Tensor grad)
        {
            var x = op.inputs[0];
            var y = op.inputs[1];

            return (grad, grad);
        }
    }
}
