using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Gradients
{
    public class array_grad
    {
        public static (Tensor, Tensor) _ReshapeGrad(Operation op, Tensor grad)
        {
            return (array_ops.reshape(grad, array_ops.shape(op.inputs[0])), null);
        }
    }
}
