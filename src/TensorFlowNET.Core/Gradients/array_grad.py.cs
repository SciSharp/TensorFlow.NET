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

        public static Tensor[] _SqueezeGrad(Operation op, Tensor[] grads)
        {
            return new Tensor[] { _ReshapeToInput(op, grads[0]) };
        }

        private static Tensor _ReshapeToInput(Operation op, Tensor grad)
        {
            return array_ops.reshape(grad, array_ops.shape(op.inputs[0]));
        }

        public static Tensor[] _TransposeGrad(Operation op, Tensor[] grads)
        {
            var p = op.inputs[1];
            return new Tensor[] { array_ops.transpose(grads[0], array_ops.invert_permutation(p)), null };
        }
    }
}
