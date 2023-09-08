/*****************************************************************************
   Copyright 2023 The TensorFlow.NET Authors. All Rights Reserved.

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

using Tensorflow.NumPy;
using Tensorflow.Operations;

namespace Tensorflow
{
    public partial class tensorflow
    {
        public MathApi math { get; } = new MathApi();
        public class MathApi
        {
            public Tensor argmax(Tensor input, Axis axis = null, string name = null, int? dimension = null, TF_DataType output_type = TF_DataType.TF_INT64)
                => gen_math_ops.arg_max(input, axis, name: name, output_type: output_type);

            public Tensor count_nonzero(Tensor input, Axis? axis = null, bool? keepdims = null, TF_DataType dtype = TF_DataType.TF_INT64, string name = null)
                => math_ops.count_nonzero_v2(input, axis: axis, keepdims: keepdims ?? false, dtype: dtype);
            public Tensor log(Tensor x, string name = null)
                => gen_math_ops.log(x, name);

            /// <summary>
            /// Computes the Gauss error function of `x` element-wise.
            /// </summary>
            /// <param name="x"></param>
            /// <param name="name"></param>
            /// <returns></returns>
            public Tensor erf(Tensor x, string name = null)
                => math_ops.erf(x, name);

            public Tensor multiply(Tensor x, Tensor y, string name = null)
                => math_ops.multiply(x, y, name: name);
            public Tensor divide_no_nan(Tensor a, Tensor b, string name = null)
                => math_ops.div_no_nan(a, b);

            /// <summary>
            /// Computes the Euclidean norm of elements across dimensions of a tensor.
            /// </summary>
            /// <param name="input_tensor">The tensor to reduce. Should have numeric type.</param>
            /// <param name="axis">The dimensions to reduce. If `None` (the default), reduces all dimensions.Must be in the range `[-rank(input_tensor), rank(input_tensor))`</param>
            /// <param name="keepdims">If true, retains reduced dimensions with length 1.</param>
            /// <param name="name">A name for the operation (optional).</param>
            /// <returns>The reduced tensor, of the same dtype as the input_tensor.</returns>
            public Tensor reduce_euclidean_norm(Tensor input_tensor, Axis? axis = null, bool keepdims = false, string name = null)
                => math_ops.reduce_euclidean_norm(input_tensor, axis: axis, keepdims: keepdims, name);

            public Tensor square(Tensor x, string name = null)
                => math_ops.square(x, name: name);

            public Tensor sum(Tensor x, Axis? axis = null, string name = null)
                => math_ops.reduce_sum(x, axis: axis, name: name);

            public Tensor softplus(Tensor features, string name = null)
                => nn_ops.softplus(features, name: name);

            public Tensor tanh(Tensor x, string name = null)
                => math_ops.tanh(x, name: name);

            /// <summary>
            /// Finds values and indices of the `k` largest entries for the last dimension.
            /// </summary>
            /// <param name="input"></param>
            /// <param name="k"></param>
            /// <param name="sorted"></param>
            /// <param name="name"></param>
            /// <returns></returns>
            public Tensors top_k(Tensor input, int k, bool sorted = true, string name = null)
                => nn_ops.top_kv2(input, k, sorted: sorted, name: name);

            public Tensor in_top_k(Tensor predictions, Tensor targets, int k, string name = "InTopK")
                => nn_ops.in_top_k(predictions, targets, k, name);

            /// <summary>
            /// 
            /// </summary>
            /// <param name="arr"></param>
            /// <param name="weights"></param>
            /// <param name="minlength"></param>
            /// <param name="maxlength"></param>
            /// <param name="dtype"></param>
            /// <param name="name"></param>
            /// <param name="axis"></param>
            /// <param name="binary_output"></param>
            /// <returns></returns>
            public Tensor bincount(Tensor arr, Tensor weights = null,
                Tensor minlength = null,
                Tensor maxlength = null,
                TF_DataType dtype = TF_DataType.TF_INT32,
                string name = null,
                Shape axis = null,
                bool binary_output = false)
                => math_ops.bincount(arr, weights: weights, minlength: minlength, maxlength: maxlength,
                    dtype: dtype, name: name, axis: axis, binary_output: binary_output);

            public Tensor real(Tensor x, string name = null)
                    => gen_ops.real(x, x.dtype.real_dtype(), name);
            public Tensor imag(Tensor x, string name = null)
                => gen_ops.imag(x, x.dtype.real_dtype(), name);

            public Tensor conj(Tensor x,  string name = null)
                => gen_ops.conj(x, name);
            public Tensor angle(Tensor x, string name = null)
                => gen_ops.angle(x, x.dtype.real_dtype(), name);
        }

        public Tensor abs(Tensor x, string name = null)
            => math_ops.abs(x, name);

        /// <summary>
        /// Computes acos of x element-wise.
        /// </summary>
        /// <param name="x"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public Tensor acos(Tensor x, string name = null)
            => gen_math_ops.acos(x, name);

        /// <summary>
        /// Computes asin of x element-wise.
        /// </summary>
        /// <param name="x"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public Tensor asin(Tensor x, string name = null)
            => gen_math_ops.asin(x, name);

        public Tensor add(Tensor a, Tensor b, string name = null)
            => gen_math_ops.add(a, b, name: name);

        public Tensor add<Tx, Ty>(Tx a, Ty b, string name = null)
            => gen_math_ops.add(ops.convert_to_tensor(a), ops.convert_to_tensor(b), name: name);

        /// <summary>
        /// Adds all input tensors element-wise.
        /// </summary>
        /// <param name="inputs"></param>
        /// <param name="name"></param>
        /// <returns>A `Tensor` of same shape and type as the elements of `inputs`.</returns>
        public Tensor add_n(Tensor[] inputs, string name = null)
            => math_ops.add_n(inputs, name: name);

        /// <summary>
        /// Computes atan of x element-wise.
        /// </summary>
        /// <param name="x"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public Tensor atan(Tensor x, string name = null)
            => gen_math_ops.atan(x, name);

        public Tensor arg_max(Tensor input, int dimension, TF_DataType output_type = TF_DataType.TF_INT64, string name = null)
            => gen_math_ops.arg_max(input, ops.convert_to_tensor(dimension), output_type: output_type, name: name);

        public Tensor arg_min(Tensor input, int dimension, TF_DataType output_type = TF_DataType.TF_INT64, string name = null)
            => gen_math_ops.arg_min(input, ops.convert_to_tensor(dimension), output_type: output_type, name: name);

        public Tensor is_finite(Tensor input, string name = null)
            => gen_math_ops.is_finite(input, name);

        public Tensor is_nan(Tensor input, string name = null)
            => gen_math_ops.is_nan(input, name);

        /// <summary>
        /// Returns element-wise smallest integer not less than x.
        /// </summary>
        /// <param name="x"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public Tensor ceil(Tensor x, string name = null)
            => gen_math_ops.ceil(x, name);

        /// <summary>
        /// Computes sin of x element-wise.
        /// </summary>
        /// <param name="x"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public Tensor sin(Tensor x, string name = null)
            => gen_math_ops.sin(x, name);

        /// <summary>
        /// Computes hyperbolic sine of x element-wise.
        /// </summary>
        /// <param name="x"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public Tensor sinh(Tensor x, string name = null)
            => gen_math_ops.sinh(x, name);

        /// <summary>
        /// Computes cos of x element-wise.
        /// </summary>
        /// <param name="x"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public Tensor cos(Tensor x, string name = null)
            => gen_math_ops.cos(x, name);

        public Tensor cos(float x, string name = null)
            => gen_math_ops.cos(ops.convert_to_tensor(x), name);

        /// <summary>
        /// Computes hyperbolic cosine of x element-wise.
        /// </summary>
        /// <param name="x"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public Tensor cosh(Tensor x, string name = null)
            => gen_math_ops.cosh(x, name);

        public Tensor tan(Tensor x, string name = null)
            => gen_math_ops.tan(x, name);

        public Tensor tanh(Tensor x, string name = null)
            => gen_math_ops.tanh(x, name);

        /// <summary>
        /// Returns element-wise largest integer not greater than x.
        /// </summary>
        /// <param name="x"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public Tensor floor(Tensor x, string name = null)
            => gen_math_ops.floor(x, name);

        /// <summary>
        /// Returns the truth value of (x > y) element-wise.
        /// </summary>
        /// <typeparam name="Tx"></typeparam>
        /// <typeparam name="Ty"></typeparam>
        /// <param name="x"></param>
        /// <param name="y"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public Tensor greater<Tx, Ty>(Tx x, Ty y, string name = null)
            => gen_math_ops.greater(ops.convert_to_tensor(x), ops.convert_to_tensor(y), name);

        /// <summary>
        /// Returns the truth value of (x >= y) element-wise.
        /// </summary>
        /// <typeparam name="Tx"></typeparam>
        /// <typeparam name="Ty"></typeparam>
        /// <param name="x"></param>
        /// <param name="y"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public Tensor greater_equal<Tx, Ty>(Tx x, Ty y, string name = null)
            => gen_math_ops.greater_equal(ops.convert_to_tensor(x), ops.convert_to_tensor(y), name);

        /// <summary>
        /// Returns the truth value of (x &lt; y) element-wise.
        /// </summary>
        /// <typeparam name="Tx"></typeparam>
        /// <typeparam name="Ty"></typeparam>
        /// <param name="x"></param>
        /// <param name="y"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public Tensor less<Tx, Ty>(Tx x, Ty y, string name = null)
            => gen_math_ops.less(ops.convert_to_tensor(x), ops.convert_to_tensor(y), name);

        /// <summary>
        /// Computes the log of the absolute value of `Gamma(x)` element-wise.
        /// </summary>
        /// <param name="x">A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`.</param>
        /// <param name="name">A name for the operation (optional).</param>
        /// <returns>A `Tensor`. Has the same type as `x`.</returns>
        public Tensor lgamma(Tensor x, string name = null)
            => gen_math_ops.lgamma(x, name: name);

        /// <summary>
        /// Returns the truth value of (x &lt;= y) element-wise.
        /// </summary>
        /// <typeparam name="Tx"></typeparam>
        /// <typeparam name="Ty"></typeparam>
        /// <param name="x"></param>
        /// <param name="y"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public Tensor less_equal<Tx, Ty>(Tx x, Ty y, string name = null)
            => gen_math_ops.less_equal(ops.convert_to_tensor(x), ops.convert_to_tensor(y), name);

        /// <summary>
        /// Computes natural logarithm of (1 + x) element-wise.
        /// </summary>
        /// <param name="x"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public Tensor log1p(Tensor x, string name = null)
            => gen_math_ops.log1p(x, name);

        public Tensor logical_and<T>(T x, T y, string name = null)
            => gen_math_ops.logical_and(ops.convert_to_tensor(x), ops.convert_to_tensor(y), name);

        public Tensor logical_not(Tensor x, string name = null)
            => gen_math_ops.logical_not(x, name);

        public Tensor logical_or(Tensor x, Tensor y, string name = null)
            => gen_math_ops.logical_or(x, y, name);

        public Tensor logical_xor(Tensor x, Tensor y, string name = "LogicalXor")
        {
            return gen_math_ops.logical_and(gen_math_ops.logical_or(x, y), 
                gen_math_ops.logical_not(gen_math_ops.logical_and(x, y)), name);
        }

        /// <summary>
        /// Clips tensor values to a specified min and max.
        /// </summary>
        /// <param name="t"></param>
        /// <param name="clip_value_min"></param>
        /// <param name="clip_value_max"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public Tensor _clip_by_value(Tensor t, Tensor clip_value_min, Tensor clip_value_max, string name = null)
            => gen_math_ops.clip_by_value(t, clip_value_min, clip_value_max);

        /// <summary>
        ///    Clips tensor values to a specified min and max.
        /// </summary>
        /// <param name="t">
        ///    A <c>Tensor</c>.
        /// </param>
        /// <param name="clip_value_min">
        ///    A 0-D (scalar) <c>Tensor</c>, or a <c>Tensor</c> with the same shape
        ///    as <c>t</c>. The minimum value to clip by.
        /// </param>
        /// <param name="clip_value_max">
        ///    A 0-D (scalar) <c>Tensor</c>, or a <c>Tensor</c> with the same shape
        ///    as <c>t</c>. The maximum value to clip by.
        /// </param>
        /// <param name="name">
        /// If specified, the created operation in the graph will be this one, otherwise it will be named 'ClipByValue'.
        /// </param>
        /// <returns>
        ///    A clipped <c>Tensor</c> with the same shape as input 't'.
        ///    The Operation can be fetched from the resulting Tensor, by fetching the Operation property from the result.
        /// </returns>
        /// <remarks>
        ///    Given a tensor <c>t</c>, this operation returns a tensor of the same type and
        ///    shape as <c>t</c> with its values clipped to <c>clip_value_min</c> and <c>clip_value_max</c>.
        ///    Any values less than <c>clip_value_min</c> are set to <c>clip_value_min</c>. Any values
        ///    greater than <c>clip_value_max</c> are set to <c>clip_value_max</c>.
        /// </remarks>
        public Tensor clip_by_value<T1, T2>(Tensor t, T1 clip_value_min, T2 clip_value_max, string name = "ClipByValue")
            => clip_ops.clip_by_value(t, clip_value_min, clip_value_max, name);

        public Tensor sub<Tx, Ty>(Tx a, Ty b, string name = null)
            => gen_math_ops.sub(ops.convert_to_tensor(a), ops.convert_to_tensor(b), name: name);

        public Tensor divide(Tensor a, Tensor b)
            => a / b;

        public Tensor sqrt(Tensor a, string name = null)
            => math_ops.sqrt(a, name);

        public Tensor sign(Tensor a, string name = null)
            => gen_math_ops.sign(a, name);

        public Tensor subtract<T>(Tensor x, T[] y, string name = null) where T : struct
            => gen_math_ops.sub(x, ops.convert_to_tensor(y, dtype: x.dtype.as_base_dtype(), name: "y"), name);

        /// <summary>
        /// return x - y
        /// </summary>
        /// <param name="x"></param>
        /// <param name="y"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public Tensor subtract(Tensor x, Tensor y, string name = null)
            => gen_math_ops.sub(x, y, name);

        public Tensor log(Tensor x, string name = null)
            => gen_math_ops.log(x, name);

        public Tensor equal(Tensor x, Tensor y, string name = null)
            => gen_math_ops.equal(x, y, name: name);

        /// <summary>
        /// Computes arctangent of `y/x` element-wise, respecting signs of the arguments.
        /// </summary>
        /// <param name="y"></param>
        /// <param name="x"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public Tensor atan2(Tensor y, Tensor x, string name = null)
            => gen_math_ops.atan2(y, x, name);

        /// <summary>
        /// Computes the maximum of elements across dimensions of a tensor.
        /// </summary>
        /// <typeparam name="Tx"></typeparam>
        /// <typeparam name="Ty"></typeparam>
        /// <param name="input"></param>
        /// <param name="axis"></param>
        /// <param name="keep_dims"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public Tensor max<Tx, Ty>(Tx input, Ty axis, bool keep_dims = false, string name = null)
            => gen_math_ops.max(ops.convert_to_tensor(input), ops.convert_to_tensor(axis), keep_dims: keep_dims, name: name);

        /// <summary>
        /// Computes the minimum of elements across dimensions of a tensor.
        /// </summary>
        /// <typeparam name="Tx"></typeparam>
        /// <typeparam name="Ty"></typeparam>
        /// <param name="input"></param>
        /// <param name="axis"></param>
        /// <param name="keep_dims"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public Tensor min<Tx, Ty>(Tx input, Ty axis, bool keep_dims = false, string name = null)
            => gen_math_ops.min(ops.convert_to_tensor(input), ops.convert_to_tensor(axis), keep_dims: keep_dims, name: name);

        /// <summary>
        /// Returns the max of x and y (i.e. x > y ? x : y) element-wise.
        /// </summary>
        /// <typeparam name="T1"></typeparam>
        /// <typeparam name="T2"></typeparam>
        /// <param name="x"></param>
        /// <param name="y"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public Tensor maximum<T1, T2>(T1 x, T2 y, string name = null)
            => gen_math_ops.maximum(ops.convert_to_tensor(x), ops.convert_to_tensor(y), name: name);

        /// <summary>
        /// Returns the min of x and y (i.e. x &lt; y ? x : y) element-wise.
        /// </summary>
        /// <typeparam name="T1"></typeparam>
        /// <typeparam name="T2"></typeparam>
        /// <param name="x"></param>
        /// <param name="y"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public Tensor minimum<T1, T2>(T1 x, T2 y, string name = null)
            => gen_math_ops.minimum(ops.convert_to_tensor(x), ops.convert_to_tensor(y), name: name);

        public Tensor multiply(Tensor x, Tensor y, string name = null)
            => gen_math_ops.mul(x, y, name: name);

        /// <summary>
        /// return x * y
        /// </summary>
        /// <typeparam name="Tx"></typeparam>
        /// <typeparam name="Ty"></typeparam>
        /// <param name="x"></param>
        /// <param name="y"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public Tensor multiply<Tx, Ty>(Tx x, Ty y, string name = null)
            => gen_math_ops.mul(ops.convert_to_tensor(x), ops.convert_to_tensor(y), name: name);
        /// <summary>
        /// return scalar product
        /// </summary>
        /// <typeparam name="Tx"></typeparam>
        /// <typeparam name="Ty"></typeparam>
        /// <param name="x"></param>
        /// <param name="y"></param>
        /// <param name="axes"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public Tensor dot_prod<Tx, Ty>(Tx x, Ty y, NDArray axes, string name = null)
            => math_ops.tensordot(convert_to_tensor(x), convert_to_tensor(y), axes, name: name);
        public Tensor negative(Tensor x, string name = null)
            => gen_math_ops.neg(x, name);

        /// <summary>
        /// Returns the truth value of (x != y) element-wise.
        /// </summary>
        /// <param name="x"></param>
        /// <param name="y"></param>
        /// <param name="name"></param>
        /// <returns>A `Tensor` of type bool with the same size as that of x or y.</returns>
        public Tensor not_equal<Tx, Ty>(Tx x, Ty y, string name = null)
            => math_ops.not_equal(x, y, name: name);

        /// <summary>
        /// Divides x / y elementwise (using Python 2 division operator semantics).
        /// </summary>
        /// <param name="x"></param>
        /// <param name="y"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public Tensor div(Tensor x, Tensor y, string name = null)
            => math_ops.div(x, y, name: name);

        public Tensor divide<T>(Tensor x, T[] y, string name = null) where T : struct
            => x / ops.convert_to_tensor(y, dtype: x.dtype.as_base_dtype(), name: "y");

        public Tensor pow<T1, T2>(T1 x, T2 y, string name = "pow")
            => math_ops.pow(x, y, name: name);

        /// <summary>
        /// Divides `x / y` elementwise, rounding toward the most negative integer.
        /// </summary>
        /// <param name="x"></param>
        /// <param name="y"></param>
        /// <param name="name"></param>
        /// <returns>`x / y` rounded down.</returns>
        public Tensor floordiv(Tensor x, Tensor y, string name = null)
            => math_ops.floordiv(x, y, name: name);

        /// <summary>
        /// Divides x / y elementwise (using Python 3 division operator semantics).
        /// </summary>
        /// <param name="x"></param>
        /// <param name="y"></param>
        /// <param name="name"></param>
        /// <returns>`x / y` evaluated in floating point.</returns>
        public static Tensor truediv(Tensor x, Tensor y, string name = null)
            => math_ops.truediv(x, y, name: name);

        public Tensor range(object start, object limit = null, object delta = null, TF_DataType? dtype = null, string name = "range")
            => math_ops.range(start, limit: limit, delta: delta, dtype: dtype, name: name);

        public Tensor real(Tensor input, string name = null)
            => math_ops.real(input, name);

        /// <summary>
        /// Computes the "logical or" of elements across dimensions of a tensor.
        /// </summary>
        /// <param name="input_tensor">The boolean tensor to reduce.</param>
        /// <param name="axis">The dimensions to reduce.</param>
        /// <param name="keepdims">If true, retains reduced dimensions with length 1.</param>
        /// <param name="name"></param>
        /// <returns>The reduced tensor.</returns>
        public Tensor reduce_any(Tensor input_tensor, Axis? axis = null, bool keepdims = false, string name = null)
            => math_ops.reduce_any(input_tensor, axis: axis, keepdims: keepdims, name: name);

        /// <summary>
        /// Computes the "logical and" of elements across dimensions of a tensor.
        /// </summary>
        /// <param name="input_tensor"></param>
        /// <param name="axis"></param>
        /// <param name="keepdims"></param>
        /// <param name="name"></param>
        /// <returns>The reduced tensor.</returns>
        public Tensor reduce_all(Tensor input_tensor, Axis? axis = null, bool keepdims = false, string name = null)
            => math_ops.reduce_all(input_tensor, axis: axis, keepdims: keepdims, name: name);

        /// <summary>
        /// Computes the product of elements across dimensions of a tensor.
        /// </summary>
        /// <param name="input_tensor"></param>
        /// <param name="axis"></param>
        /// <param name="keepdims"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public Tensor reduce_prod(Tensor input_tensor, Axis? axis = null, bool keepdims = false, string name = null)
            => math_ops.reduce_prod(input_tensor, axis: axis, keepdims: keepdims, name: name);

        /// <summary>
        /// Computes the sum of elements across dimensions of a tensor.
        /// </summary>
        /// <param name="input"></param>
        /// <param name="axis"></param>
        /// <returns></returns>
        public Tensor reduce_sum(Tensor input, Axis? axis = null, Axis? reduction_indices = null,
            bool keepdims = false, string name = null)
        {
            if (keepdims)
                return math_ops.reduce_sum(input, axis: constant_op.constant(axis ?? reduction_indices), keepdims: keepdims, name: name);
            else
                return math_ops.reduce_sum(input, axis: constant_op.constant(axis ?? reduction_indices));
        }

        /// <summary>
        /// Computes the maximum of elements across dimensions of a tensor.
        /// </summary>
        /// <param name="input_tensor"></param>
        /// <param name="axis"></param>
        /// <param name="keepdims"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public Tensor reduce_max(Tensor input_tensor, Axis? axis = null, bool keepdims = false, string name = null)
            => math_ops.reduce_max(input_tensor, axis, keepdims, name);

        public Tensor reduce_min(Tensor input_tensor, Axis? axis = null, bool keepdims = false, string name = null)
            => math_ops.reduce_min(input_tensor, axis, keepdims, name);

        public Tensor reduce_std(Tensor input_tensor, Axis? axis = null, bool keepdims = false, string name = null)
            => math_ops.reduce_std(input_tensor, axis, keepdims, name);

        public Tensor reduce_variance(Tensor input_tensor, Axis? axis = null, bool keepdims = false, string name = null)
            => math_ops.reduce_variance(input_tensor, axis, keepdims, name);

        public Tensor sigmoid<T>(T x, string name = null)
            => math_ops.sigmoid(x, name: name);

        public Tensor sum(Tensor input, int axis, bool keep_dims = false, string name = null)
            => gen_math_ops.sum(input, ops.convert_to_tensor(axis), keep_dims: keep_dims, name: name);

        public Tensor reduce_mean(Tensor input_tensor, Axis? axis = null, bool keepdims = false, string name = null, int? reduction_indices = null)
            => math_ops.reduce_mean(input_tensor, axis: axis, keepdims: keepdims, name: name, reduction_indices: reduction_indices);

        public Tensor round(Tensor x, string name = null)
            => gen_math_ops.round(x, name: name);

        public Tensor cast(Tensor x, TF_DataType dtype, string name = null)
            => math_ops.cast(x, dtype, name);

        public Tensor cumsum(Tensor x, int axis = 0, bool exclusive = false, bool reverse = false, string name = null)
            => math_ops.cumsum(x, axis: axis, exclusive: exclusive, reverse: reverse, name: name);

        public Tensor square(Tensor x, string name = null)
            => gen_math_ops.square(x, name: name);
        public Tensor squared_difference(Tensor x, Tensor y, string name = null)
            => gen_math_ops.squared_difference(x: x, y: y, name: name);
        public Tensor complex(Tensor real, Tensor imag, Tensorflow.TF_DataType? dtype = null, 
                string name = null) => gen_ops.complex(real, imag, dtype, name);
        public Tensor exp(Tensor x,
                string name = null) => gen_math_ops.exp(x, name);
    }
}
