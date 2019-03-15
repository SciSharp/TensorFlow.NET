using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Gradients
{
    public class array_grad
    {
        public static Tensor[] _ReshapeGrad(Operation op, Tensor[] grads)
        {
            return new Tensor[] { array_ops.reshape(grads[0], array_ops.shape(op.inputs[0])), null };
        }
    }
}
