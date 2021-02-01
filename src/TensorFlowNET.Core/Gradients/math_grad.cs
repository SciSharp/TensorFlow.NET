/*****************************************************************************
   Copyright 2018 The TensorFlow.NET Authors. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
******************************************************************************/

using NumSharp;
using System;
using System.Linq;
using Tensorflow.Eager;
using static Tensorflow.Binding;

namespace Tensorflow.Gradients
{
    /// <summary>
    /// Gradients for operators defined in math_ops.py.
    /// </summary>
    [RegisterGradient("math_grad")]
    public class math_grad
    {
        [RegisterGradient("Abs")]
        public static Tensor[] _AbsGrad(Operation op, Tensor[] grads)
        {
            var x = op.inputs[0];
            var grad = grads[0];

            return new Tensor[] { grad * math_ops.sign(x) };
        }

        [RegisterGradient("AddV2")]
        public static Tensor[] _AddV2Grad(Operation op, Tensor[] grads)
            => _AddGrad(op, grads);

        [RegisterGradient("Add")]
        public static Tensor[] _AddGrad(Operation op, Tensor[] grads)
        {
            var x = op.inputs[0];
            var y = op.inputs[1];
            var grad = grads[0];
            if (grad is Tensor &&
                _ShapesFullySpecifiedAndEqual(x, y, grad))
                return new Tensor[] { grad, grad };

            var sx = array_ops.shape(x);
            var sy = array_ops.shape(y);
            var (rx, ry) = gen_array_ops.broadcast_gradient_args(sx, sy);

            var sum1 = math_ops.reduce_sum(grad, rx);
            var r1 = gen_array_ops.reshape(sum1, sx);
            var sum2 = math_ops.reduce_sum(grad, ry);
            var r2 = gen_array_ops.reshape(sum2, sy);

            return new Tensor[] { r1, r2 };
        }

        /// <summary>
        /// Copies the gradient to all inputs.
        /// </summary>
        /// <param name="op"></param>
        /// <param name="grads"></param>
        /// <returns></returns>
        [RegisterGradient("AddN")]
        public static Tensor[] _AddNGrad(Operation op, Tensor[] grads)
        {
            var grad = grads[0];

            return Enumerable.Range(0, len(op.inputs))
                .Select(x => grad)
                .ToArray();
        }

        [RegisterGradient("Cumsum")]
        public static Tensor[] _CumsumGrad(Operation op, Tensor[] grads)
        {
            var grad = grads[0];
            var axis = op.inputs[1];
            var exclusive = op.get_attr<bool>("exclusive");
            var reverse = op.get_attr<bool>("reverse");
            return new Tensor[]
            {
                math_ops.cumsum(grad, axis, exclusive: exclusive, reverse: !reverse),
                null
            };
        }

        [RegisterGradient("DivNoNan")]
        public static Tensor[] _DivNoNanGrad(Operation op, Tensor[] grads)
        {
            var grad = grads[0];
            var x = op.inputs[0];
            var y = op.inputs[1];
            var sx = array_ops.shape(x);
            var sy = array_ops.shape(y);
            var (rx, ry) = gen_array_ops.broadcast_gradient_args(sx, sy);
            x = math_ops.conj(x);
            y = math_ops.conj(y);

            var reduce_sum1 = math_ops.reduce_sum(math_ops.div_no_nan(grad, y), rx);
            var reduce_sum2 = math_ops.reduce_sum(grad * math_ops.div_no_nan(math_ops.div_no_nan(-x, y), y), ry);

            return new Tensor[]
            {
                array_ops.reshape(reduce_sum1, sx),
                array_ops.reshape(reduce_sum2, sy)
            };
        }

        /// <summary>
        /// Returns grad * exp(x).
        /// </summary>
        /// <param name="op"></param>
        /// <param name="grads"></param>
        /// <returns></returns>
        [RegisterGradient("Exp")]
        public static Tensor[] _ExpGrad(Operation op, Tensor[] grads)
        {
            var grad = grads[0];
            var y = op.outputs[0];  // y = e^x
            return tf_with(ops.control_dependencies(new Operation[] { grad }), dp =>
            {
                y = math_ops.conj(y);
                // forward_compatible(2019, 9, 14)
                // return new Tensor[] { math_ops.mul_no_nan(y, grad) };
                return new Tensor[] { grad * y };
            });
        }

        [RegisterNoGradient("GreaterEqual")]
        public static Tensor[] _GreaterEqualGrad(Operation op, Tensor[] grads) => null;

        [RegisterNoGradient("OnesLike")]
        public static Tensor[] _OnesLike(Operation op, Tensor[] grads) => null;

        [RegisterNoGradient("ZerosLike")]
        public static Tensor[] _ZerosLike(Operation op, Tensor[] grads) => null;

        [RegisterGradient("Identity")]
        public static Tensor[] _IdGrad(Operation op, Tensor[] grads)
        {
            return new Tensor[] { grads[0] };
        }

        [RegisterGradient("Lgamma")]
        public static Tensor[] _LgammaGrad(Operation op, Tensor[] grads)
        {
            var grad = grads[0];
            var x = op.inputs[0];
            return tf_with(ops.control_dependencies(new Operation[] { grad }), dp =>
            {
                x = math_ops.conj(x);
                return new Tensor[] { grad * math_ops.digamma(x) };
            });
        }

        [RegisterGradient("Log")]
        public static Tensor[] _LogGrad(Operation op, Tensor[] grads)
        {
            var grad = grads[0];
            var x = op.inputs[0];
            return tf_with(ops.control_dependencies(new Operation[] { grad }), dp =>
            {
                x = math_ops.conj(x);
                return new Tensor[] { grad * math_ops.reciprocal(x) };
            });
        }

        [RegisterGradient("Log1p")]
        public static Tensor[] _Log1pGrad(Operation op, Tensor[] grads)
        {
            var grad = grads[0];
            var x = op.inputs[0];
            return tf_with(ops.control_dependencies(new Operation[] { grad }), dp =>
            {
                x = math_ops.conj(x);
                return new Tensor[] { grad * math_ops.reciprocal(1 + x) };
            });
        }

        [RegisterGradient("Mul")]
        public static Tensor[] _MulGrad(Operation op, Tensor[] grads)
        {
            var x = op.inputs[0];
            var y = op.inputs[1];
            var grad = grads[0];

            if (op is EagerOperation op_eager &&
                op_eager.SkipInputIndices.Contains(1) &&
                y.NDims == 0)
            {
                return new Tensor[]
                {
                    gen_math_ops.mul(grad, math_ops.conj(y)),
                    null
                };
            }

            if (grad is Tensor &&
                _ShapesFullySpecifiedAndEqual(x, y, grad) &&
                new TF_DataType[] { tf.int32, tf.float32 }.Contains(grad.dtype))
            {
                return new Tensor[]
                {
                    gen_math_ops.mul(grad, y),
                    gen_math_ops.mul(grad, x)
                };
            }

            var broads = SmartBroadcastGradientArgs(x, y, grad);
            var (sx, rx, must_reduce_x) = broads[0];
            var (sy, ry, must_reduce_y) = broads[1];

            x = math_ops.conj(x);
            y = math_ops.conj(y);

            Tensor gx = null, gy = null;

            if (op is EagerOperation op_eager1 &&
                op_eager1.SkipInputIndices.Contains(0))
                gy = null;
            else if (!must_reduce_x)
                gx = gen_math_ops.mul(grad, y);
            else
                gx = array_ops.reshape(
                    math_ops.reduce_sum(gen_math_ops.mul(grad, y), rx), sx);

            if (op is EagerOperation op_eager2 &&
                op_eager2.SkipInputIndices.Contains(1))
                gy = null;
            else if (!must_reduce_y)
                gy = gen_math_ops.mul(x, grad);
            else
                gy = array_ops.reshape(
                    math_ops.reduce_sum(gen_math_ops.mul(x, grad), ry), sy);

            return new Tensor[] { gx, gy };
        }

        [RegisterGradient("MatMul")]
        public static Tensor[] _MatMulGrad(Operation op, Tensor[] grads)
        {
            var grad = grads[0];
            Tensor grad_a = null, grad_b = null;

            var t_a = (bool)op.get_attr("transpose_a");
            var t_b = (bool)op.get_attr("transpose_b");
            var a = math_ops.conj(op.inputs[0]);
            var b = math_ops.conj(op.inputs[1]);
            if (!t_a && !t_b)
            {
                grad_a = gen_math_ops.mat_mul(grad, b, transpose_b: true);
                grad_b = gen_math_ops.mat_mul(a, grad, transpose_a: true);
            }
            else if (!t_a && t_b)
            {
                grad_a = gen_math_ops.mat_mul(grad, b);
                grad_b = gen_math_ops.mat_mul(grad, a, transpose_a: true);
            }
            else if (t_a && !t_b)
            {
                grad_a = gen_math_ops.mat_mul(grad, b);
                grad_b = gen_math_ops.mat_mul(grad, a, transpose_a: true);
            }
            else if (t_a && t_b)
            {
                grad_a = gen_math_ops.mat_mul(b, grad, transpose_a: true, transpose_b: true);
                grad_b = gen_math_ops.mat_mul(grad, a, transpose_a: true, transpose_b: true);
            }

            return new Tensor[] { grad_a, grad_b };
        }

        [RegisterGradient("BatchMatMul")]
        public static Tensor[] _BatchMatMul(Operation op, Tensor[] grads)
        {
            var grad = grads[0];
            Tensor grad_a = null, grad_b = null;

            var t_a = (bool)op.get_attr("adj_x");
            var t_b = (bool)op.get_attr("adj_y");
            var a = math_ops.conj(op.inputs[0]);
            var b = math_ops.conj(op.inputs[1]);
            if (!t_a && !t_b)
            {
                grad_a = gen_math_ops.batch_mat_mul(grad, b, adj_y: true);
                grad_b = gen_math_ops.batch_mat_mul(a, grad, adj_x: true);
            }
            else if (!t_a && t_b)
            {
                grad_a = gen_math_ops.batch_mat_mul(grad, b);
                grad_b = gen_math_ops.batch_mat_mul(grad, a, adj_x: true);
            }
            else if (t_a && !t_b)
            {
                grad_a = gen_math_ops.batch_mat_mul(grad, b);
                grad_b = gen_math_ops.batch_mat_mul(grad, a, adj_x: true);
            }
            else if (t_a && t_b)
            {
                grad_a = gen_math_ops.batch_mat_mul(b, grad, adj_x: true, adj_y: true);
                grad_b = gen_math_ops.batch_mat_mul(grad, a, adj_x: true, adj_y: true);
            }

            return new Tensor[] { grad_a, grad_b };
        }

        [RegisterGradient("Mean")]
        public static Tensor[] _MeanGrad(Operation op, Tensor[] grads)
        {
            var grad = grads[0];
            var sum_grad = _SumGrad(op, grads)[0];
            var input_shape = op.inputs[0]._shape_tuple();
            var output_shape = op.outputs[0]._shape_tuple();

            Tensor result, factor_tensor;
            if (tf.executing_eagerly()
                && input_shape != null
                && output_shape != null)
            {
                var input_size = np.prod(input_shape);
                var output_size = np.prod(output_shape);
                var factor = (int)input_size / Math.Max((int)output_size, 1);
                factor_tensor = constant_op.constant(factor, dtype: sum_grad.dtype);
            }
            else
            {
                var input_shape_tensor = array_ops.shape(op.inputs[0]);
                var output_shape_tensor = array_ops.shape(op.outputs[0]);
                factor_tensor = _safe_shape_div(math_ops.reduce_prod(input_shape_tensor), math_ops.reduce_prod(output_shape_tensor));
            }

            result = math_ops.truediv(sum_grad, math_ops.cast(factor_tensor, sum_grad.dtype));
            return new Tensor[] { result, null };
        }

        /// <summary>
        /// Gradient for Max.
        /// </summary>
        /// <param name="op"></param>
        /// <param name="grads"></param>
        /// <returns></returns>
        [RegisterGradient("Max")]
        public static Tensor[] _MaxGrad(Operation op, Tensor[] grads)
        {
            return _MinOrMaxGrad(op, grads);
        }

        /// <summary>
        /// Gradient for Min.
        /// </summary>
        /// <param name="op"></param>
        /// <param name="grads"></param>
        /// <returns></returns>
        [RegisterGradient("Min")]
        public static Tensor[] _MinGrad(Operation op, Tensor[] grads)
        {
            return _MinOrMaxGrad(op, grads);
        }

        private static Tensor[] _MinOrMaxGrad(Operation op, Tensor[] grads)
        {
            var grad = grads[0];
            var input_shape = array_ops.shape(op.inputs[0]);
            var output_shape_kept_dims = math_ops.reduced_shape(input_shape, op.inputs[1]);
            var y = op.outputs[0];
            y = array_ops.reshape(y, output_shape_kept_dims);
            grad = array_ops.reshape(grad, output_shape_kept_dims);

            // Compute the number of selected (maximum or minimum) elements in each
            // reduction dimension. If there are multiple minimum or maximum elements
            // then the gradient will be divided between them.
            var indicators = math_ops.cast(math_ops.equal(y, op.inputs[0]), grad.dtype);
            var num_selected = array_ops.reshape(math_ops.reduce_sum(indicators, op.inputs[1]), output_shape_kept_dims);

            return new Tensor[] { math_ops.div(indicators, num_selected) * grad, null };
        }

        /// <summary>
        /// Returns grad*(x > y, x &lt;= y) with type of grad.
        /// </summary>
        /// <param name="op"></param>
        /// <param name="grads"></param>
        /// <returns></returns>
        [RegisterGradient("Maximum")]
        public static Tensor[] _MaximumGrad(Operation op, Tensor[] grads)
        {
            return _MaximumMinimumGrad(true, op, grads[0]);
        }

        /// <summary>
        /// Returns grad*(x &lt; y, x >= y) with type of grad.
        /// </summary>
        /// <param name="op"></param>
        /// <param name="grads"></param>
        /// <returns></returns>
        [RegisterGradient("Minimum")]
        public static Tensor[] _MinimumGrad(Operation op, Tensor[] grads)
        {
            return _MaximumMinimumGrad(false, op, grads[0]);
        }

        /// <summary>
        /// Factor out the code for the gradient of Maximum or Minimum.
        /// </summary>
        /// <param name="op"></param>
        /// <param name="grad"></param>
        /// <returns></returns>
        private static Tensor[] _MaximumMinimumGrad(bool isMaximum, Operation op, Tensor grad)
        {
            var x = op.inputs[0];
            var y = op.inputs[1];
            var gdtype = grad.dtype;
            var sx = array_ops.shape(x);
            var sy = array_ops.shape(y);
            var gradshape = array_ops.shape(grad);
            var zeros = array_ops.zeros(gradshape, gdtype);
            var xmask =
                isMaximum
                    ? gen_math_ops.greater_equal(x, y)
                    : gen_math_ops.less_equal(x, y);
            var (rx, ry) = gen_array_ops.broadcast_gradient_args(sx, sy);
            var xgrad = array_ops.where(xmask, grad, zeros);
            var gx = array_ops.reshape(math_ops.reduce_sum(xgrad, rx), sx);
            var ygrad = array_ops.where(xmask, zeros, grad);
            var gy = array_ops.reshape(math_ops.reduce_sum(ygrad, ry), sy);
            return new Tensor[] { gx, gy };
        }

        [RegisterGradient("Neg")]
        public static Tensor[] _NegGrad(Operation op, Tensor[] grads)
        {
            return new Tensor[] { -grads[0] };
        }

        [RegisterGradient("Select")]
        public static Tensor[] _SelectGrad(Operation op, Tensor[] grads)
        {
            var grad = grads[0];
            var c = op.inputs[0];
            var x = op.inputs[1];
            var zeros = array_ops.zeros_like(x);
            return new Tensor[]
            {
                null,
                array_ops.where(c, grad, zeros),
                array_ops.where(c, zeros, grad)
            };
        }

        private static Tensor _safe_shape_div(Tensor x, Tensor y)
        {
            return math_ops.floordiv(x, gen_math_ops.maximum(y, 1));
        }

        [RegisterGradient("Sub")]
        public static Tensor[] _SubGrad(Operation op, Tensor[] grads)
        {
            var grad = grads[0];
            var x = op.inputs[0];
            var y = op.inputs[1];
            if (grad is Tensor &&
                _ShapesFullySpecifiedAndEqual(x, y, grad))
                return new Tensor[] { grad, -grad };

            var broads = SmartBroadcastGradientArgs(x, y, grad);
            var (sx, rx, must_reduce_x) = broads[0];
            var (sy, ry, must_reduce_y) = broads[1];

            var gx = array_ops.reshape(math_ops.reduce_sum(grad, rx), sx);
            var gy = array_ops.reshape(math_ops.reduce_sum(-grad, ry), sy);

            return new Tensor[] { gx, gy };
        }

        public static bool _ShapesFullySpecifiedAndEqual(Tensor x, Tensor y, Tensor grad)
        {
            var x_shape = x._shape_tuple();
            var y_shape = y._shape_tuple();
            var grad_shape = grad._shape_tuple();
            return x_shape != null &&
                y_shape != null &&
                Enumerable.SequenceEqual(x_shape, y_shape) &&
                Enumerable.SequenceEqual(y_shape, grad_shape) &&
                !x_shape.Contains(-1);
        }

        [RegisterGradient("Sum")]
        public static Tensor[] _SumGrad(Operation op, Tensor[] grads)
        {
            var grad = grads[0];
            var input_0_shape = op.inputs[0]._shape_tuple();
            Tensor input_shape = null;

            if (input_0_shape != null)
            {
                var axes = tensor_util.constant_value(op.inputs[1]);
                if (!(axes is null))
                {
                    var rank = input_0_shape.Length;
                    if (Enumerable.SequenceEqual(Enumerable.Range(0, rank), axes.Data<int>()))
                    {
                        if (tf.Context.executing_eagerly())
                        {
                            // should add ones_rank_cache
                            var new_shape = constant_op.constant(range(0, rank).Select(x => 1).ToArray(), dtype: TF_DataType.TF_INT32);
                            grad = array_ops.reshape(grad, new_shape);
                        }
                        else
                        {
                            var new_shape = range(rank).Select(x => 1).ToArray();
                            grad = array_ops.reshape(grad, new_shape);
                        }

                        // If shape is not fully defined (but rank is), we use Shape.
                        if (!input_0_shape.Contains(-1))
                            input_shape = constant_op.constant(input_0_shape);
                        else
                            input_shape = array_ops.shape(op.inputs[0]);
                        return new Tensor[] { gen_array_ops.tile(grad, input_shape), null };
                    }
                    else if (!input_0_shape.Contains(-1) && !tf.Context.executing_eagerly())
                    {
                        throw new NotImplementedException("");
                    }
                }
            }

            input_shape = array_ops.shape(op.inputs[0]);

            if (tf.executing_eagerly())
            {
                if (!op.get_attr<bool>("keep_dims"))
                {
                    ops.colocate_with(input_shape);
                    var output_shape_kept_dims = math_ops.reduced_shape(input_shape, op.inputs[1]);
                    // var tile_scaling = _safe_shape_div(input_shape, output_shape_kept_dims);
                    grad = gen_array_ops.reshape(grad, output_shape_kept_dims);
                }

                return new Tensor[] { gen_array_ops.broadcast_to(grad, input_shape), null };
            }
            else
            {
                ops.colocate_with(input_shape);
                var output_shape_kept_dims = math_ops.reduced_shape(input_shape, op.inputs[1]);
                var tile_scaling = _safe_shape_div(input_shape, output_shape_kept_dims);
                grad = gen_array_ops.reshape(grad, output_shape_kept_dims);

                return new Tensor[] { gen_array_ops.tile(grad, tile_scaling), null };
            }
        }

        [RegisterGradient("RealDiv")]
        public static Tensor[] _RealDivGrad(Operation op, Tensor[] grads)
        {
            var grad = grads[0];
            var x = op.inputs[0];
            var y = op.inputs[1];

            var sx = array_ops.shape(x);
            var sy = array_ops.shape(y);
            var (rx, ry) = gen_array_ops.broadcast_gradient_args(sx, sy);
            x = math_ops.conj(x);
            y = math_ops.conj(y);

            var reshape1 = array_ops.reshape(
                math_ops.reduce_sum(
                    math_ops.realdiv(grad, y), rx),
                sx);
            var reshape2 = array_ops.reshape(
                math_ops.reduce_sum(
                    grad * math_ops.realdiv(math_ops.realdiv(-x, y), y), ry),
                sy);

            return new Tensor[] { reshape1, reshape2 };
        }

        [RegisterGradient("Sigmoid")]
        public static Tensor[] _SigmoidGrad(Operation op, Tensor[] grads)
        {
            var grad = grads[0];
            var y = op.outputs[0];

            return tf_with(ops.control_dependencies(grads), delegate
            {
                y = math_ops.conj(y);
                return new Tensor[] { gen_math_ops.sigmoid_grad(y, grad) };
            });
        }

        [RegisterGradient("Sign")]
        public static Tensor[] _SignGrad(Operation op, Tensor[] grads)
        {
            var x = op.inputs[0];
            var zero = constant_op.constant(0.0f, x.dtype, x.shape);

            return new Tensor[] { zero };
        }

        [RegisterGradient("Square")]
        public static Tensor[] _SquareGrad(Operation op, Tensor[] grads)
        {
            var grad = grads[0];
            var x = op.inputs[0];

            return tf_with(ops.control_dependencies(grads), delegate
            {
                x = math_ops.conj(x);
                var y = constant_op.constant(2.0, dtype: x.dtype);
                return new Tensor[] { math_ops.multiply(grad, math_ops.multiply(x, y)) };
            });
        }

        [RegisterGradient("Sqrt")]
        public static Tensor[] _SqrtGrad(Operation op, Tensor[] grads)
        {
            var grad = grads[0];
            var y = op.outputs[0];

            return tf_with(ops.control_dependencies(grads), delegate
            {
                y = math_ops.conj(y);
                var factor = constant_op.constant(0.5f, dtype: y.dtype);
                return new Tensor[] { grad * (factor * math_ops.reciprocal(y)) };
            });
        }

        [RegisterGradient("Sin")]
        public static Tensor[] _SinGrad(Operation op, Tensor[] grads)
        {
            var grad = grads[0];
            var x = op.inputs[0];

            return tf_with(ops.control_dependencies(grads), delegate
            {
                x = math_ops.conj(x);
                return new Tensor[] { math_ops.multiply(grad, gen_math_ops.cos(x)) };
            });
        }

        [RegisterGradient("Sinh")]
        public static Tensor[] _SinhGrad(Operation op, Tensor[] grads)
        {
            var grad = grads[0];
            var x = op.inputs[0];

            return tf_with(ops.control_dependencies(grads), delegate
            {
                x = math_ops.conj(x);
                return new Tensor[] { math_ops.multiply(grad, gen_math_ops.cosh(x)) };
            });
        }

        [RegisterGradient("Cos")]
        public static Tensor[] _CosGrad(Operation op, Tensor[] grads)
        {
            var grad = grads[0];
            var x = op.inputs[0];

            return tf_with(ops.control_dependencies(grads), delegate
            {
                x = math_ops.conj(x);
                return new Tensor[] { math_ops.multiply(grad, -gen_math_ops.sin(x)) };
            });
        }

        [RegisterGradient("Cosh")]
        public static Tensor[] _CoshGrad(Operation op, Tensor[] grads)
        {
            var grad = grads[0];
            var x = op.inputs[0];

            return tf_with(ops.control_dependencies(grads), delegate
            {
                x = math_ops.conj(x);
                return new Tensor[] { math_ops.multiply(grad, gen_math_ops.sinh(x)) };
            });
        }

        [RegisterGradient("Tanh")]
        public static Tensor[] _TanhGrad(Operation op, Tensor[] grads)
        {
            var grad = grads[0];
            var y = op.outputs[0];

            return tf_with(ops.control_dependencies(grads), delegate
            {
                y = math_ops.conj(y);
                return new Tensor[] { gen_math_ops.tanh_grad(y, grad) };
            });
        }

        [RegisterGradient("Pow")]
        public static Tensor[] _PowGrad(Operation op, Tensor[] grads)
        {
            var grad = grads[0];
            var x = op.inputs[0];
            var y = op.inputs[1];

            if (op is EagerOperation op_eager &&
                op_eager.SkipInputIndices.Contains(1) &&
                y.NDims == 0)
            {
                x = math_ops.conj(x);
                y = math_ops.conj(y);
                return new Tensor[]
                {
                    grad * y * math_ops.pow(x, y - 1),
                    null
                };
            }

            var z = op.outputs[0];

            var broads = SmartBroadcastGradientArgs(x, y, grad);
            var (sx, rx, must_reduce_x) = broads[0];
            var (sy, ry, must_reduce_y) = broads[1];

            x = math_ops.conj(x);
            y = math_ops.conj(y);
            z = math_ops.conj(z);
            var mul = grad * y * math_ops.pow(x, y - 1.0f);
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

            return new Tensor[] { gx, gy };
        }

        /// <summary>
        /// Optimized version of `broadcast_gradient_args` that caches results.
        /// </summary>
        /// <param name="x"></param>
        /// <param name="y"></param>
        /// <returns></returns>
        private static (Tensor, Tensor, bool)[] SmartBroadcastGradientArgs(Tensor x, Tensor y, Tensor grad)
        {
            Tensor sx, sy;
            if (x.TensorShape.is_fully_defined() &&
                y.TensorShape.is_fully_defined())
            {
                sx = array_ops.shape(x);
                sy = array_ops.shape(y);
            }
            else
            {
                sx = array_ops.shape_internal(x, optimize: false);
                sy = array_ops.shape_internal(y, optimize: false);
            }

            var (rx, ry) = gen_array_ops.broadcast_gradient_args(sx, sy);
            return new[]
            {
                (sx, rx, !x.TensorShape.Equals(grad.TensorShape)),
                (sy, ry, !y.TensorShape.Equals(grad.TensorShape))
            };
        }
    }
}
