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

using System;
using System.Linq;
using Tensorflow.Contexts;
using static Tensorflow.Binding;

namespace Tensorflow
{
    public static partial class gen_math_ops
    {
        public static Tensor _all(Tensor input, Tensor axis, bool keep_dims = false, string name = null)
        {
            var _op = tf.OpDefLib._apply_op_helper("All", name, args: new { input, reduction_indices = axis, keep_dims = keep_dims });

            return _op.outputs[0];
        }

        /// <summary>
        /// Add all input tensors element wise.
        /// </summary>
        /// <param name="inputs"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public static Tensor add_n(Tensor[] inputs, string name = null)
            => tf.Context.ExecuteOp("AddN", name, new ExecuteOpArgs()
            {
                OpInputArgs = new object[] { inputs }
            });

        /// <summary>
        /// Returns the index with the largest value across dimensions of a tensor.
        /// </summary>
        /// <param name="input"></param>
        /// <param name="dimension"></param>
        /// <param name="output_type"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public static Tensor arg_max(Tensor input, int dimension, TF_DataType output_type = TF_DataType.TF_INT64, string name = null)
            => tf.Context.ExecuteOp("ArgMax", name, new ExecuteOpArgs(input, dimension)
                .SetAttributes(new { output_type }));


        /// <summary>
        /// Returns the index with the smallest value across dimensions of a tensor.
        /// </summary>
        /// <param name="input"></param>
        /// <param name="dimension"></param>
        /// <param name="output_type"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public static Tensor arg_min(Tensor input, int dimension, TF_DataType output_type = TF_DataType.TF_INT64, string name = null)
            => tf.Context.ExecuteOp("ArgMin", name, new ExecuteOpArgs(input, dimension)
                .SetAttributes(new { output_type }));

        /// <summary>
        /// Computes Psi, the derivative of Lgamma (the log of the absolute value of
        /// `Gamma(x)`), element-wise.
        /// </summary>
        /// <param name="x"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public static Tensor digamma(Tensor x, string name = null)
            => tf.OpDefLib._apply_op_helper("Digamma", name, args: new { x }).output;

        /// <summary>
        ///    Returns 0 if the denominator is zero.
        /// </summary>
        /// <param name="x">
        /// </param>
        /// <param name="y">
        /// </param>
        /// <param name="name">
        /// If specified, the created operation in the graph will be this one, otherwise it will be named 'DivNoNan'.
        /// </param>
        /// <returns>
        ///    The Operation can be fetched from the resulting Tensor, by fetching the Operation property from the result.
        /// </returns>
        /// <remarks>
        ///    
        ///    *NOTE*: <c>DivNoNan</c> supports broadcasting. More about broadcasting
        ///    [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
        /// </remarks>
        public static Tensor div_no_nan(Tensor x, Tensor y, string name = null)
            => tf.Context.ExecuteOp("DivNoNan", name, new ExecuteOpArgs(x, y));

        public static Tensor mean(Tensor input, int axis, bool keep_dims = false, string name = null)
            => mean(input, ops.convert_to_tensor(axis), keep_dims: keep_dims, name: name);

        /// <summary>
        /// Computes the mean of elements across dimensions of a tensor.
        /// Reduces `input` along the dimensions given in `axis`. Unless
        /// `keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
        /// `axis`. If `keep_dims` is true, the reduced dimensions are retained with length 1.
        /// </summary>
        /// <param name="input">A `Tensor`. Must be one of the following types: 
        /// `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`. 
        /// The tensor to reduce.</param>
        /// <param name="axis">A `Tensor`. Must be one of the following types: `int32`, `int64`. The dimensions to reduce.</param>
        /// <param name="keep_dims"> An optional `bool`. Defaults to `False`. If true, retain reduced dimensions with length 1.</param>
        /// <param name="name"> A name for the operation (optional).</param>
        /// <returns> A `Tensor`. Has the same type as `input`.</returns>
        public static Tensor mean(Tensor input, Tensor axis, bool keep_dims = false, string name = null)
            => tf.Context.ExecuteOp("Mean", name, new ExecuteOpArgs(input, axis)
            {
                GetGradientAttrs = (op) => new
                {
                    T = op.get_attr<TF_DataType>("T"),
                    Tidx = op.get_attr<TF_DataType>("Tidx"),
                    keep_dims = op.get_attr<bool>("keep_dims")
                }
            }.SetAttributes(new { keep_dims, reduction_indices = axis }));

        public static Tensor mean(Tensor[] inputs, Tensor axis, bool keep_dims = false, string name = null)
        {
            if (tf.Context.executing_eagerly())
            {
                return mean_eager_fallback(inputs, axis, keep_dims: keep_dims, name: name, ctx: tf.Context);
            }

            var _op = tf.OpDefLib._apply_op_helper("Mean", name, args: new { inputs, reduction_indices = axis, keep_dims = keep_dims });

            return _op.output;
        }

        private static Tensor mean_eager_fallback(Tensor[] inputs, Tensor axis, bool keep_dims = false, string name = null, Context ctx = null)
        {
            var (_attr_T, input) = tf.Runner.ArgsToMatchingEager(ctx, args: new[] { inputs });
            var (_attr_Tidx, axis1) = tf.Runner.ArgsToMatchingEager(ctx, default_dtype: tf.int32, args: new[] { axis });
            var _inputs_flat = input.concat(axis1);
            var _attrs = new object[] { "keep_dims", keep_dims, "T", _attr_T, "Tidx", _attr_Tidx };

            return tf.Runner.Execute(ctx, "Mean", 1, _inputs_flat, _attrs, name: name)[0];
        }

        public static Tensor prod<T1, T2>(T1 input, T2 axis, bool keep_dims = false, string name = null)
            => tf.Context.ExecuteOp("Prod", name, 
                new ExecuteOpArgs(input, axis).SetAttributes(new { keep_dims, reduction_indices = axis }));

        private static Tensor prod_eager_fallback(Tensor input_t, int[] axis, bool keep_dims, string name, Context ctx = null)
        {
            var (_attr_T, input) = tf.Runner.ArgsToMatchingEager(ctx, args: new[] { input_t });
            var (_attr_Tidx, axis1) = tf.Runner.ArgsToMatchingEager(ctx, default_dtype: tf.int32, args: new[] { axis });
            var _inputs_flat = input.concat(axis1);
            var _attrs = new object[] { "keep_dims", keep_dims, "T", _attr_T, "Tidx", _attr_Tidx };

            return tf.Runner.Execute(ctx, "Prod", 1, _inputs_flat, _attrs, name: name)[0];
        }

        public static Tensor acos(Tensor x, string name = null)
            => tf.Context.ExecuteOp("Acos", name, new ExecuteOpArgs(x));

        public static Tensor asin(Tensor x, string name = null)
            => tf.Context.ExecuteOp("Asin", name, new ExecuteOpArgs(x));

        public static Tensor add(Tensor x, Tensor y, string name = null)
            => tf.Context.ExecuteOp("Add", name, new ExecuteOpArgs(x, y));

        public static Tensor add<Tx, Ty>(Tx x, Ty y, string name = null)
            => tf.Context.ExecuteOp("Add", name, new ExecuteOpArgs(x, y));

        public static Tensor add_v2<Tx, Ty>(Tx x, Ty y, string name = null)
            => tf.Context.ExecuteOp("AddV2", name, new ExecuteOpArgs(x, y));

        public static Tensor atan(Tensor x, string name = null)
            => tf.Context.ExecuteOp("Atan", name, new ExecuteOpArgs(x));

        public static Tensor ceil(Tensor x, string name = null)
            => tf.Context.ExecuteOp("Ceil", name, new ExecuteOpArgs(x));

        public static Tensor sin(Tensor x, string name = null)
            => tf.Context.ExecuteOp("Sin", name, new ExecuteOpArgs(x));

        /// <summary>
        ///    Computes sigmoid of <c>x</c> element-wise.
        /// </summary>
        /// <param name="x">
        /// </param>
        /// <param name="name">
        /// If specified, the created operation in the graph will be this one, otherwise it will be named 'Sigmoid'.
        /// </param>
        /// <returns>
        ///    The Operation can be fetched from the resulting Tensor, by fetching the Operation property from the result.
        /// </returns>
        /// <remarks>
        ///    Specifically, <c>y = 1 / (1 + exp(-x))</c>.
        /// </remarks>
        public static Tensor sigmoid(Tensor x, string name = "Sigmoid")
            => tf.Context.ExecuteOp("Sigmoid", name, new ExecuteOpArgs(x));

        /// <summary>
        ///    Computes the gradient of the sigmoid of <c>x</c> wrt its input.
        /// </summary>
        /// <param name="y">
        /// </param>
        /// <param name="dy">
        /// </param>
        /// <param name="name">
        /// If specified, the created operation in the graph will be this one, otherwise it will be named 'SigmoidGrad'.
        /// </param>
        /// <returns>
        ///    The Operation can be fetched from the resulting Tensor, by fetching the Operation property from the result.
        /// </returns>
        /// <remarks>
        ///    Specifically, <c>grad = dy * y * (1 - y)</c>, where <c>y = sigmoid(x)</c>, and
        ///    <c>dy</c> is the corresponding input gradient.
        /// </remarks>
        public static Tensor sigmoid_grad(Tensor y, Tensor dy, string name = "SigmoidGrad")
            => tf.Context.ExecuteOp("SigmoidGrad", name, new ExecuteOpArgs(y, dy));

        public static Tensor sign<T>(T x, string name = "Sign")
            => tf.Context.ExecuteOp("Sign", name, new ExecuteOpArgs(x));

        public static Tensor sinh(Tensor x, string name = null)
            => tf.Context.ExecuteOp("Sinh", name, new ExecuteOpArgs(x));

        public static Tensor cos<T>(T x, string name = null)
            => tf.Context.ExecuteOp("Cos", name, new ExecuteOpArgs(x));

        public static Tensor cosh(Tensor x, string name = null)
            => tf.Context.ExecuteOp("Cosh", name, new ExecuteOpArgs(x));

        /// <summary>
        /// Computes the sum along segments of a tensor.
        /// </summary>
        /// <param name="data"></param>
        /// <param name="segment_ids"></param>
        /// <param name="num_segments"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public static Tensor unsorted_segment_sum(Tensor data, Tensor segment_ids, Tensor num_segments, string name = null)
        {
            var _op = tf.OpDefLib._apply_op_helper("UnsortedSegmentSum", name, new { data, segment_ids, num_segments });
            return _op.outputs[0];
        }

        public static Tensor tan(Tensor x, string name = null)
            => tf.Context.ExecuteOp("Tan", name, new ExecuteOpArgs(x));

        public static Tensor tanh(Tensor x, string name = null)
            => tf.Context.ExecuteOp("Tanh", name, new ExecuteOpArgs(x));

        /// <summary>
        /// Computes the gradient for the tanh of `x` wrt its input.
        /// </summary>
        /// <param name="y"></param>
        /// <param name="dy"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public static Tensor tanh_grad(Tensor y, Tensor dy, string name = null)
            => tf.Context.ExecuteOp("TanhGrad", name, new ExecuteOpArgs(y, dy));

        public static Tensor floor(Tensor x, string name = null)
        {
            var _op = tf.OpDefLib._apply_op_helper("Floor", name, args: new { x });

            return _op.outputs[0];
        }

        public static Tensor _clip_by_value(Tensor t, Tensor clip_value_min, Tensor clip_value_max, string name = null)
        {
            var _op = tf.OpDefLib._apply_op_helper("ClipByValue", name, args: new { t, clip_value_min, clip_value_max });

            return _op.outputs[0];
        }

        public static Tensor greater<Tx, Ty>(Tx x, Ty y, string name = null)
            => tf.Context.ExecuteOp("Greater", name, new ExecuteOpArgs(x, y));

        /// <summary>
        /// Computes the log of the absolute value of `Gamma(x)` element-wise.
        /// </summary>
        /// <param name="x">
        /// A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`.
        /// </param>
        /// <param name="name">
        /// </param>
        /// <returns>
        /// The Operation can be fetched from the resulting Tensor, by fetching the Operation property from the result.
        /// </returns>
        public static Tensor lgamma(Tensor x, string name = null)
            => tf.Context.ExecuteOp("Lgamma", name, new ExecuteOpArgs(x));


        public static Tensor greater_equal<Tx, Ty>(Tx x, Ty y, string name = null)
            => tf.Context.ExecuteOp("GreaterEqual", name, new ExecuteOpArgs(x, y));

        public static Tensor less<Tx, Ty>(Tx x, Ty y, string name = null)
            => tf.Context.ExecuteOp("Less", name, new ExecuteOpArgs(x, y));

        public static Tensor less_equal<Tx, Ty>(Tx x, Ty y, string name = null)
            => tf.Context.ExecuteOp("LessEqual", name, new ExecuteOpArgs(x, y));

        public static Tensor log1p(Tensor x, string name = null)
            => tf.Context.ExecuteOp("Log1p", name, new ExecuteOpArgs(x));

        public static Tensor logical_and(Tensor x, Tensor y, string name = null)
            => tf.Context.ExecuteOp("LogicalAnd", name, new ExecuteOpArgs(x, y));

        public static Tensor logical_and(bool x, bool y, string name = null)
            => tf.Context.ExecuteOp("LogicalAnd", name, new ExecuteOpArgs(x, y));

        public static Tensor logical_not(Tensor x, string name = null)
            => tf.Context.ExecuteOp("LogicalNot", name, new ExecuteOpArgs(x));

        public static Tensor logical_or(Tensor x, Tensor y, string name = null)
            => tf.Context.ExecuteOp("LogicalOr", name, new ExecuteOpArgs(x, y));

        public static Tensor logical_xor(Tensor x, Tensor y, string name = "LogicalXor")
        {
            return logical_and(
                logical_or(x, y),
                logical_not(logical_and(x, y)),
                name);
        }

        public static Tensor squared_difference(Tensor x, Tensor y, string name = null)
            => tf.Context.ExecuteOp("SquaredDifference", name, new ExecuteOpArgs(x, y));

        /// <summary>
        /// Computes square of x element-wise.
        /// </summary>
        /// <param name="x"> A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`, `int32`, `int64`, `complex64`, `complex128`.</param>
        /// <param name="name"> A name for the operation (optional).</param>
        /// <returns> A `Tensor`. Has the same type as `x`.</returns>
        public static Tensor square(Tensor x, string name = null)
            => tf.Context.ExecuteOp("Square", name, new ExecuteOpArgs(x));

        /// <summary>
        /// Returns which elements of x are finite.
        /// </summary>
        /// <param name="x"> A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`.</param>
        /// <param name="name"> A name for the operation (optional).</param>
        /// <returns> A `Tensor` of type `bool`.</returns>
        public static Tensor is_finite(Tensor x, string name = null)
            => tf.Context.ExecuteOp("IsFinite", name, new ExecuteOpArgs(x));

        public static Tensor is_nan(Tensor x, string name = null)
            => tf.Context.ExecuteOp("IsNan", name, new ExecuteOpArgs(x));


        /// <summary>
        /// Computes exponential of x element-wise.  \\(y = e^x\\).
        /// </summary>
        /// <param name="x"> A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`, `complex64`, `complex128`.</param>
        /// <param name="name"> A name for the operation (optional).</param>
        /// <returns> A `Tensor`. Has the same type as `x`.</returns>
        public static Tensor exp(Tensor x, string name = null)
            => tf.Context.ExecuteOp("Exp", name, new ExecuteOpArgs(x));

        /// <summary>
        /// Computes natural logarithm of x element-wise.
        /// </summary>
        /// <param name="x"> A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`, `complex64`, `complex128`.</param>
        /// <param name="name"> name: A name for the operation (optional).</param>
        /// <returns> A `Tensor`. Has the same type as `x`.</returns>
        public static Tensor log(Tensor x, string name = null)
            => tf.Context.ExecuteOp("Log", name, new ExecuteOpArgs(x));

        public static Tensor softplus(Tensor features, string name = null)
            => tf.Context.ExecuteOp("Softplus", name, new ExecuteOpArgs(features));
        
        public static Tensor cast(Tensor x, TF_DataType DstT, bool Truncate = false, string name = null)
            => tf.Context.ExecuteOp("Cast", name, new ExecuteOpArgs(x)
                .SetAttributes(new { DstT, Truncate }));

        public static Tensor neg(Tensor x, string name = null)
            => tf.Context.ExecuteOp("Neg", name, new ExecuteOpArgs(x));

        public static Tensor sqrt(Tensor x, string name = null)
            => tf.Context.ExecuteOp("Sqrt", name, new ExecuteOpArgs(x));

        public static Tensor sub(Tensor x, Tensor y, string name = null)
            => tf.Context.ExecuteOp("Sub", name, new ExecuteOpArgs(x, y));

        public static Tensor sub<Tx, Ty>(Tx x, Ty y, string name = null)
            => tf.Context.ExecuteOp("Sub", name, new ExecuteOpArgs(x, y));

        /// <summary>
        /// Returns the truth value of (x == y) element-wise.
        /// </summary>
        /// <param name="x"></param>
        /// <param name="y"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public static Tensor equal<Tx, Ty>(Tx x, Ty y, string name = null)
            => tf.Context.ExecuteOp("Equal", name, new ExecuteOpArgs(x, y));

        /// <summary>
        /// Returns the truth value of (x != y) element-wise.
        /// </summary>
        /// <typeparam name="Tx">The type of the x.</typeparam>
        /// <typeparam name="Ty">The type of the y.</typeparam>
        /// <param name="x">The x.</param>
        /// <param name="y">The y.</param>
        /// <param name="name">The name.</param>
        /// <returns></returns>
        public static Tensor not_equal<Tx, Ty>(Tx x, Ty y, string name = null)
            => tf.Context.ExecuteOp("NotEqual", name, new ExecuteOpArgs(x, y));

        public static Tensor atan2(Tensor y, Tensor x, string name = null)
            => tf.Context.ExecuteOp("Atan2", name, new ExecuteOpArgs(y, x));

        public static Tensor mul<Tx, Ty>(Tx x, Ty y, string name = null)
            => tf.Context.ExecuteOp("Mul", name, new ExecuteOpArgs(x, y));

        public static Tensor mul_no_nan<Tx, Ty>(Tx x, Ty y, string name = null)
        {
            var _op = tf.OpDefLib._apply_op_helper("MulNoNan", name, args: new { x, y });

            return _op.outputs[0];
        }

        public static Tensor real_div(Tensor x, Tensor y, string name = null)
            => tf.Context.ExecuteOp("RealDiv", name, new ExecuteOpArgs(x, y));

        public static Tensor reciprocal(Tensor x, string name = null)
            => tf.Context.ExecuteOp("Reciprocal", name, new ExecuteOpArgs(x));

        public static Tensor floor_mod(Tensor x, Tensor y, string name = null)
            => tf.Context.ExecuteOp("FloorMod", name, new ExecuteOpArgs(x, y));

        public static Tensor floor_div(Tensor x, Tensor y, string name = null)
            => tf.Context.ExecuteOp("FloorDiv", name, new ExecuteOpArgs(x, y));

        /// <summary>
        /// Multiply the matrix "a" by the matrix "b".
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <param name="transpose_a"></param>
        /// <param name="transpose_b"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public static Tensor mat_mul(Tensor a, Tensor b, bool transpose_a = false, bool transpose_b = false, string name = null)
            => tf.Context.ExecuteOp("MatMul", name, new ExecuteOpArgs(a, b)
                .SetAttributes(new
                {
                    transpose_a,
                    transpose_b
                }));

        /// <summary>
        /// Returns the max of x and y (i.e. x > y ? x : y) element-wise.
        /// </summary>
        /// <param name="x"></param>
        /// <param name="y"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public static Tensor maximum<T1, T2>(T1 x, T2 y, string name = null)
            => tf.Context.ExecuteOp("Maximum", name, new ExecuteOpArgs(x, y));

        public static Tensor minimum<T1, T2>(T1 x, T2 y, string name = null)
            => tf.Context.ExecuteOp("Minimum", name, new ExecuteOpArgs(x, y));

        public static Tensor _abs(Tensor x, string name = null)
            => tf.Context.ExecuteOp("Abs", name, new ExecuteOpArgs(x));

        public static Tensor _any<Tx, Ty>(Tx input, Ty axis, bool keep_dims = false, string name = null)
        {
            var _op = tf.OpDefLib._apply_op_helper("Any", name, new { input, reduction_indices = axis, keep_dims });

            return _op.outputs[0];
        }

        /// <summary>
        /// Subroutine for Min or Max functions. See _min and _max
        /// </summary>
        private static Tensor MinOrMax<Tx, Ty>(Tx input, Ty axis, string methodName, bool keep_dims = false, string name = null)
            => tf.Context.ExecuteOp(methodName, name, new ExecuteOpArgs(input, axis)
            {
                GetGradientAttrs = (op) => new
                {
                    T = op.get_attr<TF_DataType>("T"),
                    align_corners = op.get_attr<bool>("align_corners"),
                    half_pixel_centers = op.get_attr<bool>("half_pixel_centers")
                }
            }.SetAttributes(new { keep_dims, reduction_indices = axis }));

        public static Tensor _max<Tx, Ty>(Tx input, Ty axis, bool keep_dims = false, string name = null)
            => MinOrMax(input, axis, "Max", keep_dims: keep_dims, name: name);

        public static Tensor _min<Tx, Ty>(Tx input, Ty axis, bool keep_dims = false, string name = null)
            => MinOrMax(input, axis, "Min", keep_dims: keep_dims, name: name);


        public static Tensor pow<Tx, Ty>(Tx x, Ty y, string name = null)
            => tf.Context.ExecuteOp("Pow", name, new ExecuteOpArgs(x, y));

        public static Tensor _sum<Tx, Ty>(Tx input, Ty axis = default, bool keep_dims = false, string name = null)
            => tf.Context.ExecuteOp("Sum", name, 
                new ExecuteOpArgs(input, axis).SetAttributes(new { keep_dims, reduction_indices = axis }));

        public static Tensor _sum(Tensor[] inputs, Tensor axis = default, bool keep_dims = false, string name = null)
        {
            if (tf.Context.executing_eagerly())
            {
                return _sum_eager_fallback(inputs, axis,
                        keep_dims: keep_dims, name: name, ctx: tf.Context);
            }

            var _op = tf.OpDefLib._apply_op_helper("Sum", name, args: new { inputs, reduction_indices = axis, keep_dims });

            return _op.outputs[0];
        }

        private static Tensor _sum_eager_fallback(Tensor[] inputs, Tensor axis, bool keep_dims = false, string name = null, Context ctx = null)
        {
            var (_attr_T, input) = tf.Runner.ArgsToMatchingEager(ctx, args: new[] { inputs });
            var (_attr_Tidx, axis1) = tf.Runner.ArgsToMatchingEager(ctx, tf.int32, new[] { axis });
            var _inputs_flat = input.concat(axis1);
            var _attrs = new object[] { "keep_dims", keep_dims, "T", _attr_T, "Tidx", _attr_Tidx };

            return tf.Runner.Execute(ctx, "Sum", 1, _inputs_flat, _attrs, name: name)[0];
        }

        /// <summary>
        /// Creates a sequence of numbers.
        /// </summary>
        /// <param name="start"></param>
        /// <param name="limit"></param>
        /// <param name="delta"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public static Tensor range(Tensor start, Tensor limit, Tensor delta, string name = null)
            => tf.Context.ExecuteOp("Range", name, new ExecuteOpArgs(start, limit, delta));

        /// <summary>
        ///    Rounds the values of a tensor to the nearest integer, element-wise.
        /// </summary>
        /// <param name="x">
        /// </param>
        /// <param name="name">
        /// If specified, the created operation in the graph will be this one, otherwise it will be named 'Round'.
        /// </param>
        /// <returns>
        ///    The Operation can be fetched from the resulting Tensor, by fetching the Operation property from the result.
        /// </returns>
        /// <remarks>
        ///    Rounds half to even.  Also known as bankers rounding. If you want to round
        ///    according to the current system rounding mode use std::cint.
        /// </remarks>
        public static Tensor round(Tensor x, string name = "Round")
            => tf.Context.ExecuteOp("Round", name, new ExecuteOpArgs(x));

        /// <summary>
        /// Computes reciprocal of square root of x element-wise.
        /// </summary>
        /// <param name="x"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public static Tensor rsqrt(Tensor x, string name = null)
            => tf.Context.ExecuteOp("Rsqrt", name, new ExecuteOpArgs(x));

        /// <summary>
        /// Returns the fraction of zeros in value.
        /// </summary>
        /// <param name="value">A tensor of numeric type.</param>
        /// <param name="name">A name for the operation (optional).</param>
        /// <returns>The fraction of zeros in value, with type float32.</returns>
        public static Tensor zero_fraction(Tensor value, string name = null)
            => tf.Context.ExecuteOp("zero_fraction", name, new ExecuteOpArgs(value));
    }
}
