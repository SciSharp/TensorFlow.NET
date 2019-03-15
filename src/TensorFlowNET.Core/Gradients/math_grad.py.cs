using System;
using System.Collections.Generic;
using System.Linq;
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
            if (grad is Tensor && _ShapesFullySpecifiedAndEqual(x, y, grad))
                return (grad, grad);

            var sx = array_ops.shape(x);
            var sy = array_ops.shape(y);
            var (rx, ry) = gen_array_ops.broadcast_gradient_args(sx, sy);

            var r1 = gen_array_ops.reshape(math_ops.reduce_sum(grad, rx), sx);
            var r2 = gen_array_ops.reshape(math_ops.reduce_sum(grad, ry), sy);

            return (r1, r2);
        }

        public static Tensor _IdGrad(Operation op, Tensor grad)
        {
            return grad;
        }

        public static (Tensor, Tensor) _MulGrad(Operation op, Tensor grad)
        {
            var x = op.inputs[0];
            var y = op.inputs[1];
            if (grad is Tensor && _ShapesFullySpecifiedAndEqual(x, y, grad) &&
                new TF_DataType[] { tf.int32, tf.float32 }.Contains(grad.dtype))
                return (gen_math_ops.mul(grad, y), gen_math_ops.mul(grad, x));

            var sx = array_ops.shape(x);
            var sy = array_ops.shape(y);
            var (rx, ry) = gen_array_ops.broadcast_gradient_args(sx, sy);

            x = math_ops.conj(x);
            y = math_ops.conj(y);

            var mul1 = gen_math_ops.mul(grad, y);
            var mul2 = gen_math_ops.mul(x, grad);
            var reduce_sum1 = math_ops.reduce_sum(mul1, rx);
            var reduce_sum2 = math_ops.reduce_sum(mul2, ry);
            var reshape1 = gen_array_ops.reshape(reduce_sum1, sx);
            var reshape2 = gen_array_ops.reshape(reduce_sum2, sy);

            return (reshape1, reshape2);
        }

        public static (Tensor, Tensor) _MeanGrad(Operation op, Tensor grad)
        {
            var sum_grad = _SumGrad(op, grad).Item1;
            var input_shape = op.inputs[0]._shape_tuple();
            var output_shape = op.outputs[0]._shape_tuple();

            var input_shape_tensor = array_ops.shape(op.inputs[0]);
            var output_shape_tensor = array_ops.shape(op.outputs[0]);
            var factor = _safe_shape_div(math_ops.reduce_prod(input_shape_tensor), math_ops.reduce_prod(output_shape_tensor));

            return (math_ops.truediv(sum_grad, math_ops.cast(factor, sum_grad.dtype)), null);
        }

        private static Tensor _safe_shape_div(Tensor x, Tensor y)
        {
            return math_ops.floordiv(x, gen_math_ops.maximum(y, 1));
        }

        public static (Tensor, Tensor) _SubGrad(Operation op, Tensor grad)
        {
            var x = op.inputs[0];
            var y = op.inputs[1];
            if (grad is Tensor && _ShapesFullySpecifiedAndEqual(x, y, grad))
                return (grad, -grad);

            var sx = array_ops.shape(x);
            var sy = array_ops.shape(y);
            var (rx, ry) = gen_array_ops.broadcast_gradient_args(sx, sy);

            var r1 = gen_array_ops.reshape(math_ops.reduce_sum(grad, rx), sx);
            var r2 = gen_array_ops.reshape(-math_ops.reduce_sum(grad, ry), sy);

            return (r1, r2);
        }

        public static bool _ShapesFullySpecifiedAndEqual(Tensor x, Tensor y, Tensor grad)
        {
            return x.NDims == y.NDims && y.NDims == grad.NDims && x.NDims > -1;
        }

        public static (Tensor, Tensor) _SumGrad(Operation op, Tensor grad)
        {
            var input_0_shape = op.inputs[0]._shape_tuple();
            Tensor input_shape = null;

            if (input_0_shape != null)
            {
                var axes = tensor_util.constant_value(op.inputs[1]);
                if(!(axes is null))
                {
                    var rank = axes.shape.Rank;
                    grad = array_ops.reshape(grad, new int[] { 1 });
                    if (!input_0_shape.Contains(-1))
                        input_shape = constant_op.constant(input_0_shape);
                    else
                        input_shape = array_ops.shape(op.inputs[0]);
                    return (gen_array_ops.tile(grad, input_shape), null);
                }
            }

            input_shape = array_ops.shape(op.inputs[0]);
            ops.colocate_with(input_shape);
            var output_shape_kept_dims = math_ops.reduced_shape(input_shape, op.inputs[1]);
            var tile_scaling = _safe_shape_div(input_shape, output_shape_kept_dims);
            grad = gen_array_ops.reshape(grad, output_shape_kept_dims);

            return (gen_array_ops.tile(grad, tile_scaling), null);
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

            var realdiv1 = gen_math_ops.real_div(-x, y);
            var realdiv2 = gen_math_ops.real_div(realdiv1, y);
            var reduce_sum1 = math_ops.reduce_sum(grad * realdiv2, ry);
            var reshape1 = gen_array_ops.reshape(reduce_sum1, sy);
            var realdiv3 = gen_math_ops.real_div(grad, y);
            var reduce_sum2 = math_ops.reduce_sum(realdiv3, rx);
            var reshape2 = gen_array_ops.reshape(reduce_sum2, sx);

            return (reshape2, reshape1);
        }

        public static (Tensor, Tensor) _PowGrad(Operation op, Tensor grad)
        {
            var x = op.inputs[0];
            var y = op.inputs[1];
            var z = op.outputs[0];

            var sx = array_ops.shape(x);
            var sy = array_ops.shape(y);
            var (rx, ry) = gen_array_ops.broadcast_gradient_args(sx, sy);
            x = math_ops.conj(x);
            y = math_ops.conj(y);
            z = math_ops.conj(z);
            var pow = gen_math_ops.pow(x, y - 1.0f);
            var mul = grad * y * pow;
            var reduce_sum = math_ops.reduce_sum(mul, rx);
            var gx = gen_array_ops.reshape(reduce_sum, sx);

            // Avoid false singularity at x = 0
            Tensor mask = null;
            if (x.dtype.is_complex())
                throw new NotImplementedException("x.dtype.is_complex()");
            else
                mask = x > 0.0f;
            var ones = array_ops.ones_like(x);
            var safe_x = array_ops.where(mask, x, ones);
            var x1 = gen_array_ops.log(safe_x);
            var y1 = array_ops.zeros_like(x);
            var log_x = array_ops.where(mask, x1, y1);
            var mul1 = grad * z * log_x;
            var reduce_sum1 = math_ops.reduce_sum(mul1, ry);
            var gy = gen_array_ops.reshape(reduce_sum1, sy);

            return (gx, gy);
        }
    }
}
