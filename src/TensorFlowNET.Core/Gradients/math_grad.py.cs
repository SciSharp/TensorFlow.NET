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

        public static (Tensor, Tensor) _SumGrad(Operation op, Tensor grad)
        {
            if (op.inputs[0].NDims > -1)
            {

            }

            var input_shape = array_ops.shape(op.inputs[0]);
            ops.colocate_with(input_shape);
            var output_shape_kept_dims = math_ops.reduced_shape(input_shape, op.inputs[1]);
            //var tile_scaling = _safe_shape_div(input_shape, output_shape_kept_dims);
            //var grad = array_ops.reshape(grad, output_shape_kept_dims);

            return (grad, grad);
        }

        public static (Tensor, Tensor) _RealDivGrad(Operation op, Tensor grad)
        {
            var x = op.inputs[0];
            var y = op.inputs[1];

            var sx = array_ops.shape(x);
            var sy = array_ops.shape(y);
            var (rx, ry) = gen_array_ops.broadcast_gradient_args(sx, sy);
            x = math_ops.conj(x);
            y = math_ops.conj(y);

            var realdiv1 = gen_math_ops.real_div(grad, y);
            var reduce_sum1 = math_ops.reduce_sum(realdiv1, rx);
            var realdiv2 = gen_math_ops.real_div(-x, y);
            var realdiv3 = gen_math_ops.real_div(realdiv2, y);
            var reduce_sum2 = math_ops.reduce_sum(grad * realdiv3, ry);

            return (gen_array_ops.reshape(reduce_sum1, sx), gen_array_ops.reshape(reduce_sum2, sy));
        }
    }
}
