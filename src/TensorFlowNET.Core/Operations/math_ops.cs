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
using System.Collections.Generic;
using System.Linq;
using Tensorflow.Framework;
using static Tensorflow.Binding;

namespace Tensorflow
{
    /// <summary>
    /// python\ops\math_ops.py
    /// </summary>
    public class math_ops
    {
        public static Tensor abs(Tensor x, string name = null)
        {
            return tf_with(ops.name_scope(name, "Abs", new { x }), scope =>
            {
                name = scope;
                x = ops.convert_to_tensor(x, name: "x");
                if (x.dtype.is_complex())
                    throw new NotImplementedException("math_ops.abs for dtype.is_complex");
                //return gen_math_ops.complex_abs(x, Tout: x.dtype.real_dtype, name: name);
                return gen_math_ops._abs(x, name: name);
            });
        }

        public static Tensor add<Tx, Ty>(Tx x, Ty y, string name = null)
            => gen_math_ops.add(x, y, name);

        public static Tensor add_v2(Tensor x, Tensor y, string name = null)
            => tf.Context.ExecuteOp("AddV2", name, new ExecuteOpArgs(x, y));

        public static Tensor add_v2<Tx, Ty>(Tx x, Ty y, string name = null)
            => gen_math_ops.add_v2(x, y, name);

        /// <summary>
        /// Adds all input tensors element-wise.
        /// </summary>
        /// <param name="inputs"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public static Tensor add_n(Tensor[] inputs, string name = null)
        {
            inputs = ops.convert_n_to_tensor_or_indexed_slices(inputs);

            if (inputs.Length == 1)
            {
                var values = inputs[0];
                if (name != null)
                    return array_ops.identity(values, name: name);
                return values;
            }

            return gen_math_ops.add_n(inputs, name: name);
        }

        public static Tensor argmax(Tensor input, int dimension, TF_DataType output_type = TF_DataType.TF_INT64, string name = null)
            => gen_math_ops.arg_max(input, dimension, output_type: output_type, name: name);

        public static Tensor round(Tensor x, string name = null)
        {
            x = ops.convert_to_tensor(x, name: "x");
            if (x.dtype.is_integer())
                return x;
            else
                return gen_math_ops.round(x, name: name);
        }

        public static Tensor cast(IVariableV1 x, TF_DataType dtype = TF_DataType.DtInvalid, string name = null)
        {
            var base_type = dtype.as_base_dtype();
            if (base_type == x.dtype)
                return x.AsTensor();

            return tf_with(ops.name_scope(name, "Cast", new { x }), scope =>
            {
                name = scope;
                var t_x = ops.convert_to_tensor(x, name: "x");
                if (t_x.dtype.as_base_dtype() != base_type)
                    t_x = gen_math_ops.cast(t_x, base_type, name: name);

                return x.AsTensor();
            });
        }

        public static ResourceVariable cast(ResourceVariable x, TF_DataType dtype = TF_DataType.DtInvalid, string name = null)
        {
            var base_type = dtype.as_base_dtype();
            if (base_type == x.dtype)
                return x;

            return tf_with(ops.name_scope(name, "Cast", new { x }), scope =>
            {
                name = scope;
                var t_x = ops.convert_to_tensor(x, name: "x");
                if (t_x.dtype.as_base_dtype() != base_type)
                    t_x = gen_math_ops.cast(t_x, base_type, name: name);

                return x;
            });
        }

        public static Tensor cast(Tensor x, TF_DataType dtype = TF_DataType.DtInvalid, string name = null)
        {
            var base_type = dtype.as_base_dtype();
            if (base_type == x.dtype)
                return x;

            return tf_with(ops.name_scope(name, "Cast", new { x }), scope =>
            {
                name = scope;
                x = ops.convert_to_tensor(x, name: "x");
                if (x.dtype.as_base_dtype() != base_type)
                    x = gen_math_ops.cast(x, base_type, name: name);

                return x;
            });
        }

        public static Tensor saturate_cast(Tensor value, TF_DataType dtype, string name = null)
        {
            return tf_with(ops.name_scope(name, "saturate_cast", new[] { value }), name =>
             {
                 value = ops.convert_to_tensor(value, name: "value");
                 // dtype = dtypes.as_dtype(dtype).as_base_dtype();
                 if (value.dtype.min() < dtype.min())
                     value = gen_math_ops.maximum(
                         value,
                         ops.convert_to_tensor(dtype.min(), dtype: value.dtype, name: "min"));
                 if (value.dtype.max() > dtype.max())
                     value = gen_math_ops.minimum(
                         value,
                         ops.convert_to_tensor(dtype.max(), dtype: value.dtype, name: "max"));
                 return cast(value, dtype, name: name);
             });
        }

        public static Tensor cast(float x, TF_DataType dtype = TF_DataType.DtInvalid, string name = null)
        {
            var base_type = dtype.as_base_dtype();

            return tf_with(ops.name_scope(name, "Cast", new { x }), scope =>
            {
                name = scope;
                var x_tensor = ops.convert_to_tensor(x, name: "x");
                if (x_tensor.dtype.as_base_dtype() != base_type)
                    x_tensor = gen_math_ops.cast(x_tensor, base_type, name: name);

                return x_tensor;
            });
        }

        public static Tensor cumsum<T>(Tensor x, T axis = default, bool exclusive = false, bool reverse = false, string name = null)
            => tf_with(ops.name_scope(name, "Cumsum", new { x }), scope =>
            {
                name = scope;
                return tf.Context.ExecuteOp("Cumsum", name, new ExecuteOpArgs(x, axis)
                    .SetAttributes(new { exclusive, reverse }));
            });

        /// <summary>
        /// Computes Psi, the derivative of Lgamma (the log of the absolute value of
        /// `Gamma(x)`), element-wise.
        /// </summary>
        /// <param name="x"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public static Tensor digamma(Tensor x, string name = null)
            => gen_math_ops.digamma(x, name: name);

        /// <summary>
        /// Divide two values using Python 2 semantics. Used for Tensor.__div__.
        /// </summary>
        /// <param name="x">`Tensor` numerator of real numeric type.</param>
        /// <param name="y">`Tensor` denominator of real numeric type.</param>
        /// <param name="name">A name for the operation</param>
        /// <returns>`x / y` returns the quotient of x and y.</returns>
        public static Tensor div(Tensor x, Tensor y, string name = null)
        {
            return tf_with(ops.name_scope(name, "div", (x, y)), name_scope =>
            {
                name = name_scope;
                x = ops.convert_to_tensor(x, name: "x");
                y = ops.convert_to_tensor(y, dtype: x.dtype.as_base_dtype(), name = "y");
                var x_dtype = x.dtype.as_base_dtype();
                var y_dtype = y.dtype.as_base_dtype();
                if (x_dtype != y_dtype)
                    throw new TypeError($"x and y must have the same dtype, got {x_dtype} != {y_dtype}");
                if (x_dtype.is_floating() || x_dtype.is_complex())
                    return gen_math_ops.real_div(x, y, name: name);
                else
                    return gen_math_ops.floor_div(x, y, name: name);
            });
        }

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
        {
            return tf_with(ops.name_scope(name, "div_no_nan", (x, y)), name_scope =>
            {
                name = name_scope;
                x = ops.convert_to_tensor(x, name: "x");
                y = ops.convert_to_tensor(y, name: "y", dtype: x.dtype.as_base_dtype());
                var x_dtype = x.dtype.as_base_dtype();
                var y_dtype = y.dtype.as_base_dtype();
                if (x_dtype != y_dtype)
                    throw new TypeError($"x and y must have the same dtype, got {x_dtype} != {y_dtype}");
                return gen_math_ops.div_no_nan(x, y, name: name);
            });
        }

        public static Tensor greater_equal<Tx, Ty>(Tx x, Ty y, string name = null)
            => gen_math_ops.greater_equal<Tx, Ty>(x, y, name: name);
        public static Tensor equal<Tx, Ty>(Tx x, Ty y, string name = null)
            => gen_math_ops.equal(x, y, name: name);

        /// <summary>
        /// Computes the Gauss error function of `x` element-wise.
        /// </summary>
        /// <param name="x"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public static Tensor erf(Tensor x, string name = null)
            => tf.Context.ExecuteOp("Erf", name, new ExecuteOpArgs(x));

        public static Tensor sqrt(Tensor x, string name = null)
            => gen_math_ops.sqrt(x, name: name);

        public static Tensor multiply(Tensor x, Tensor y, string name = null)
            => tf.Context.ExecuteOp("Mul", name, new ExecuteOpArgs(x, y));

        public static Tensor multiply<Tx, Ty>(Tx x, Ty y, string name = null)
            => gen_math_ops.mul(x, y, name: name);

        public static Tensor not_equal<Tx, Ty>(Tx x, Ty y, string name = null)
            => gen_math_ops.not_equal(x, y, name: name);

        public static Tensor mul_no_nan<Tx, Ty>(Tx x, Ty y, string name = null)
            => gen_math_ops.mul_no_nan(x, y, name: name);

        public static Tensor scalar_mul<Tscale, Tx>(Tscale scale, Tx x, string name = null)
            => tf.Context.ExecuteOp("Mul", name, new ExecuteOpArgs(scale, x));

        public static Tensor real(Tensor input, string name = null)
        {
            return tf_with(ops.name_scope(name, "Real", new[] { input }), scope =>
             {
                 // name = scope;
                 input = ops.convert_to_tensor(input, name: "input");
                 if (input.dtype.is_complex())
                 {
                     var real_dtype = input.dtype.real_dtype();
                     return real(input, name: scope);
                 }
                 else
                 {
                     return input;
                 }
             });
        }

        /// <summary>
        /// Computes the mean of elements across dimensions of a tensor.
        /// Reduces `input_tensor` along the dimensions given in `axis`.
        /// Unless `keepdims` is true, the rank of the tensor is reduced by 1 for each
        /// entry in `axis`. If `keepdims` is true, the reduced dimensionsare retained with length 1.
        /// If `axis` is None, all dimensions are reduced, and a tensor with a single element is returned.
        /// </summary>
        /// <param name="input_tensor"> The tensor to reduce. Should have numeric type.</param>
        /// <param name="axis">The dimensions to reduce. If `None` (the default), reduces all
        /// dimensions.Must be in the range `[-rank(input_tensor), rank(input_tensor))`.</param>
        /// <param name="keepdims"> If true, retains reduced dimensions with length 1.</param>
        /// <param name="name"> A name for the operation (optional).</param>
        public static Tensor reduce_mean(Tensor input_tensor, int[] axis = null, bool keepdims = false, string name = null, int? reduction_indices = null)
        {
            var r = _ReductionDims(input_tensor, axis);
            var axis_tensor = axis == null ? r : ops.convert_to_tensor(axis);
            var m = gen_math_ops.mean(input_tensor, axis_tensor, keepdims, name);
            return _may_reduce_to_scalar(keepdims, axis_tensor, m);
        }

        public static Tensor reduce_mean(Tensor[] input_tensors, int? axis = null, bool keepdims = false, string name = null)
        {
            var r = _ReductionDims(input_tensors, axis);
            var axis_tensor = axis == null ? r : ops.convert_to_tensor(axis.Value);
            var m = gen_math_ops.mean(input_tensors, axis_tensor, keepdims, name);
            return _may_reduce_to_scalar(keepdims, axis, m);
        }

        /// <summary>
        /// Computes the product of elements across dimensions of a tensor.
        /// </summary>
        /// <param name="input_tensor"></param>
        /// <param name="axis"></param>
        /// <param name="keepdims"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public static Tensor reduce_prod(Tensor input_tensor, int[] axis = null, bool keepdims = false, string name = null)
        {
            var r = _ReductionDims(input_tensor, axis);
            if (axis == null)
            {
                var m = gen_math_ops.prod(input_tensor, r, keepdims, name);
                return _may_reduce_to_scalar(keepdims, axis, m);
            }
            else
            {
                var m = gen_math_ops.prod(input_tensor, axis, keepdims, name);
                return _may_reduce_to_scalar(keepdims, axis, m);
            }
        }

        public static Tensor reduce_std(Tensor input_tensor, int[] axis = null, bool keepdims = false, string name = null)
        {
            if (name == null)
                name = "reduce_std";
            // else {name = name;}

            return tf_with(ops.name_scope(name, "reduce_std", new[] { input_tensor }), scope =>
             {
                 var variance = reduce_variance(input_tensor, axis: axis, keepdims: keepdims);
                 return gen_math_ops.sqrt(variance);
             });
        }

        public static Tensor reduce_variance(Tensor input_tensor, int[] axis = null, bool keepdims = false, string name = null)
        {
            if (name == null)
                name = "reduce_variance";
            // else {name = name;}

            return tf_with(ops.name_scope(name, "reduce_variance", new[] { input_tensor }), scope =>
             {
                 var means = reduce_mean(input_tensor, axis: axis, keepdims: true);
                 if (means.dtype.is_integer())
                     throw new TypeError("Input must be either real or complex");
                 var diff = input_tensor - means;

                 Tensor squared_deviations;
                 if (diff.dtype.is_complex())
                 {
                     var real_dtype = diff.dtype.real_dtype();
                     squared_deviations = real(
                         gen_math_ops.mul(conj(diff), diff));
                 }
                 else
                 {
                     squared_deviations = gen_math_ops.square(diff);
                 }
                 return reduce_mean(squared_deviations, axis: axis, keepdims: keepdims);
             });
        }

        public static Tensor sigmoid<T>(T x, string name = null)
            => tf_with(ops.name_scope(name, "Sigmoid", x), scope =>
            {
                name = scope;
                var x_tensor = ops.convert_to_tensor(x, name: "x");
                return gen_math_ops.sigmoid(x_tensor, name: name);
            });

        public static Tensor sign<T>(T x, string name = null)
            => gen_math_ops.sign(x, name: name);

        /// <summary>
        /// Returns (x - y)(x - y) element-wise.
        /// </summary>
        /// <param name="x"> A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`, `int32`, `int64`, `complex64`, `complex128`.</param>
        /// <param name="y"> A `Tensor`. Must have the same type as `x`.</param>
        /// <param name="name"> A name for the operation (optional).</param>
        /// <returns>A `Tensor`. Has the same type as `x`.</returns>
        public static Tensor square_difference(Tensor x, Tensor y, string name = null)
        {
            var m = gen_math_ops.squared_difference(x, y);
            return m;
        }

        public static Tensor square(Tensor x, string name = null)
        {
            return gen_math_ops.square(x, name);
        }

        public static Tensor subtract<Tx, Ty>(Tx x, Ty y, string name = null)
        {
            return gen_math_ops.sub(x, y, name);
        }

        public static Tensor log(Tensor x, string name = null)
        {
            return gen_math_ops.log(x, name);
        }

        public static Tensor logical_and(Tensor x, Tensor y, string name = null)
            => gen_math_ops.logical_and(x, y, name: name);

        public static Tensor lgamma(Tensor x, string name = null)
            => gen_math_ops.lgamma(x, name: name);

        /// <summary>
        /// Helper function for reduction ops.
        /// </summary>
        /// <param name="input_shape">1-D Tensor, the shape of the Tensor being reduced.</param>
        /// <param name="axes">1-D Tensor, the reduction axes.</param>
        /// <returns>A 1-D Tensor, the output shape as if keepdims were set to True.</returns>
        public static Tensor reduced_shape(Tensor input_shape, Tensor axes)
        {
            if (tf.Context.executing_eagerly())
            {
                var input_shape_val = input_shape.numpy();
                foreach (var axes_val in axes.ToArray<int>())
                    input_shape_val[axes_val] = 1;
                return tf.constant(input_shape_val);
            }

            input_shape = to_int32(input_shape);
            axes = to_int32(axes);

            var input_rank = array_ops.size(input_shape);
            axes = (axes + input_rank) % input_rank;
            var axes_shape = array_ops.shape(axes);
            var rng = math_ops.range(input_rank);
            var a1 = new Tensor[] { rng, axes };
            var fill = gen_array_ops.fill(axes_shape, 1);
            var a2 = new Tensor[] { input_shape, fill };

            return gen_data_flow_ops.dynamic_stitch(a1, a2);
        }

        /// <summary>
        /// Computes the reciprocal of x element-wise.
        /// </summary>
        /// <param name="x"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public static Tensor reciprocal(Tensor x, string name = null)
            => gen_math_ops.reciprocal(x, name: name);

        /// <summary>
        /// Computes the "logical and" of elements across dimensions of a tensor.
        /// </summary>
        /// <param name="input_tensor"></param>
        /// <param name="axis"></param>
        /// <param name="keepdims"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public static Tensor reduce_all(Tensor input_tensor, int[] axis = null, bool keepdims = false, string name = null)
        {
            var all = gen_math_ops._all(input_tensor,
                    _ReductionDims(input_tensor, axis),
                    keepdims,
                    name: name);

            return _may_reduce_to_scalar(keepdims, axis, all);
        }

        public static Tensor realdiv(Tensor x, Tensor y, string name = null)
            => gen_math_ops.real_div(x, y, name: name);

        /// <summary>
        /// Computes log(sum(exp(elements across dimensions of a tensor))).
        /// Reduces `input_tensor` along the dimensions given in `axis`.
        /// Unless `keepdims` is true, the rank of the tensor is reduced by 1 for each
        /// entry in `axis`. If `keepdims` is true, the reduced dimensions
        /// are retained with length 1.
        ///
        /// If `axis` has no entries, all dimensions are reduced, and a
        /// tensor with a single element is returned.
        ///
        /// This function is more numerically stable than log(sum(exp(input))). It avoids
        /// overflows caused by taking the exp of large inputs and underflows caused by
        /// taking the log of small inputs.
        /// </summary>
        /// <param name="input_tensor"> The tensor to reduce. Should have numeric type.</param>
        /// <param name="axis"> The dimensions to reduce. If `None` (the default), reduces all 
        /// dimensions.Must be in the range `[-rank(input_tensor), rank(input_tensor))`.</param>
        /// <param name="keepdims"></param>
        /// <returns> The reduced tensor.</returns>
        public static Tensor reduce_logsumexp(Tensor input_tensor, int[] axis = null, bool keepdims = false, string name = null)
        {
            return tf_with(ops.name_scope(name, "ReduceLogSumExp", new { input_tensor }), scope =>
            {
                var raw_max = reduce_max(input_tensor, axis, true);
                var my_max = array_ops.stop_gradient(array_ops.where(gen_math_ops.is_finite(raw_max), raw_max, array_ops.zeros_like(raw_max)));
                var result = gen_math_ops.log(
                reduce_sum(
                    gen_math_ops.exp(gen_math_ops.sub(input_tensor, my_max)),
                    axis[0],
                    keepdims));
                if (!keepdims)
                {
                    my_max = array_ops.reshape(my_max, array_ops.shape(result));
                }
                result = gen_math_ops.add(result, my_max);
                return _may_reduce_to_scalar(keepdims, axis, result);
            });
        }

        public static Tensor reduce_any(Tensor input_tensor, int[] axis = null, bool keepdims = false, string name = null)
        {
            var r = _ReductionDims(input_tensor, axis);
            var max = (axis != null) ? gen_math_ops._any(input_tensor, axis, keepdims, name) :
                gen_math_ops._any(input_tensor, r, keepdims, name);
            return _may_reduce_to_scalar(keepdims, axis, max);
        }

        public static Tensor reduce_max(Tensor input_tensor, int[] axis = null, bool keepdims = false, string name = null)
        {
            var r = _ReductionDims(input_tensor, axis);
            var max = (axis != null) ? gen_math_ops._max(input_tensor, axis, keepdims, name) :
                gen_math_ops._max(input_tensor, r, keepdims, name);
            return _may_reduce_to_scalar(keepdims, axis, max);
        }

        public static Tensor reduce_max(Tensor input_tensor, int axis, bool keepdims = false, string name = null)
        {
            var r = _ReductionDims(input_tensor, axis);
            var max = gen_math_ops._max(input_tensor, r, keepdims, name);
            return _may_reduce_to_scalar(keepdims, axis, max);
        }

        public static Tensor reduce_min(Tensor input_tensor, int[] axis = null, bool keepdims = false, string name = null)
        {
            var r = _ReductionDims(input_tensor, axis);
            var min = gen_math_ops._min(input_tensor, r, keepdims, name);
            return _may_reduce_to_scalar(keepdims, axis, min);
        }

        /// <summary>
        /// Computes the sum along segments of a tensor.
        /// </summary>
        /// <param name="data"></param>
        /// <param name="segment_ids"></param>
        /// <param name="num_segments"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public static Tensor unsorted_segment_sum(Tensor data, Tensor segment_ids, Tensor num_segments, string name = null)
            => gen_math_ops.unsorted_segment_sum(data, segment_ids, num_segments, name: name);

        /// <summary>
        /// Casts a tensor to type `int32`.
        /// </summary>
        /// <param name="x">A `Tensor` or `SparseTensor` or `IndexedSlices`.</param>
        /// <param name="name">A name for the operation (optional).</param>
        /// <returns>A `Tensor` or `SparseTensor` or `IndexedSlices` with same shape as `x` with type `int32`.</returns>
        private static Tensor to_int32(Tensor x, string name = "ToInt32")
        {
            return __case__(x, TF_DataType.TF_INT32, name: name);
        }

        /// <summary>
        /// Casts a tensor to a new type.
        /// </summary>
        /// <param name="x"></param>
        /// <param name="dtype"></param>
        /// <param name="name"></param>
        /// <returns>A `Tensor` or `SparseTensor` or `IndexedSlices` with same shape as `x` and same type as `dtype`.</returns>
        public static Tensor __case__(Tensor x, TF_DataType dtype, string name = null)
        {
            var base_type = dtype.as_base_dtype();
            if (x is Tensor && base_type == x.dtype)
                return x;

            // math_ops.py cast
            throw new NotImplementedException();
        }

        public static Tensor reduce_sum(Tensor[] input_tensors, int? axis = null, bool keepdims = false, string name = null)
        {
            var dims = _ReductionDims(input_tensors, axis);
            var m = gen_math_ops._sum(input_tensors, dims, keep_dims: keepdims, name: name);
            return _may_reduce_to_scalar(keepdims, axis, m);
        }

        public static Tensor reduce_sum(Tensor input_tensor, Tensor axis = null, bool keepdims = false, string name = null)
        {
            var r = _ReductionDims(input_tensor, axis);
            var m = gen_math_ops._sum(input_tensor, r, keep_dims: keepdims, name: name);
            return _may_reduce_to_scalar(keepdims, axis, m);
        }

        public static Tensor reduce_sum(Tensor input_tensor, int[] axis, bool keepdims = false, string name = null)
        {
            var m = gen_math_ops._sum(input_tensor, axis, keep_dims: keepdims, name: name);
            return _may_reduce_to_scalar(keepdims, axis, m);
        }

        public static Tensor reduce_sum(Tensor input_tensor, int axis, bool keepdims = false, string name = null)
        {
            var dims = _ReductionDims(input_tensor, axis);
            var m = gen_math_ops._sum(input_tensor, dims, keep_dims: keepdims, name: name);
            return _may_reduce_to_scalar(keepdims, axis, m);
        }

        private static Tensor _may_reduce_to_scalar(bool keepdims, Tensor axis, Tensor output)
        {
            if (!common_shapes.has_fully_defined_shape(output) &&
                !keepdims &&
                axis == null)
                // We want set_shape to be reflected in the C API graph for when we run it.
                output.shape = new int[0];
            return output;
        }

        private static Tensor _may_reduce_to_scalar(bool keepdims, int[] axis, Tensor output)
        {
            if (!common_shapes.has_fully_defined_shape(output) &&
                !keepdims &&
                axis == null)
                output.shape = new int[0];
            return output;
        }

        private static Tensor _may_reduce_to_scalar(bool keepdims, int? axis, Tensor output)
        {
            return output;
        }

        private static Tensor _ReductionDims(Tensor x, Tensor axis)
        {
            if (axis != null)
            {
                return axis;
            }
            else
            {
                if (x.rank > -1 && tf.executing_eagerly())
                    return constant_op.constant(np.arange(x.rank));

                var rank = array_ops.rank(x);
                return range(0, rank, 1);
            }
        }

        private static int _ReductionDims(Tensor x, int axis)
        {
            return axis;
        }

        private static Tensor _ReductionDims(Tensor[] x, int? axis = null, string name = null)
        {
            return range(0, array_ops.rank(x));
        }

        private static Tensor _ReductionDims(Tensor x, int[] axis)
        {
            if (axis != null)
            {
                // should return axis. or check before.
                return ops.convert_to_tensor(axis, TF_DataType.TF_INT32);
            }
            else
            {
                var rank = common_shapes.rank(x);

                // we rely on Range and Rank to do the right thing at run-time.
                if (rank == -1) return range(0, array_ops.rank(x));

                if (rank.HasValue && rank.Value > -1)
                {
                    return constant_op.constant(np.arange(rank.Value), TF_DataType.TF_INT32);
                }

                return range(0, rank, 1);
            }
        }

        /// <summary>
        /// Computes reciprocal of square root of x element-wise.
        /// </summary>
        /// <param name="x"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public static Tensor rsqrt(Tensor x, string name = null)
            => gen_math_ops.rsqrt(x, name: name);

        public static Tensor pow<Tx, Ty>(Tx x, Ty y, string name = null)
            => tf_with(ops.name_scope(name, "Pow", new { x, y }), scope =>
            {
                name = scope;
                var x_tensor = ops.convert_to_tensor(x, name: "x");
                var y_tensor = ops.convert_to_tensor(y, name: "y", dtype: x_tensor.dtype.as_base_dtype());

                return tf.Context.ExecuteOp("Pow", name, new ExecuteOpArgs(x_tensor, y_tensor));
            });

        public static Tensor range(object start, object limit = null, object delta = null, TF_DataType dtype = TF_DataType.DtInvalid, string name = "range")
        {
            if (limit == null)
            {
                limit = start;
                start = 0;
            }

            if (delta == null)
                delta = 1;

            return tf_with(ops.name_scope(name, "Range", new { start, limit, delta }), scope =>
            {
                name = scope;
                var start1 = ops.convert_to_tensor(start, name: "start", dtype: dtype);
                var limit1 = ops.convert_to_tensor(limit, name: "limit", dtype: dtype);
                var delta1 = ops.convert_to_tensor(delta, name: "delta", dtype: dtype);

                return gen_math_ops.range(start1, limit1, delta1, name);
            });
        }

        public static Tensor floordiv(Tensor x, Tensor y, string name = null)
        {
            return tf_with(ops.name_scope(name, "floordiv", new { x, y }), scope =>
            {
                return gen_math_ops.floor_div(x, y, scope);
            });
        }

        public static Tensor minimum<Tx, Ty>(Tx x, Ty y, string name = null)
            => gen_math_ops.minimum(x, y, name: name);

        public static Tensor maximum<Tx, Ty>(Tx x, Ty y, string name = null)
            => gen_math_ops.maximum(x, y, name: name);

        /// <summary>
        /// Multiplies matrix `a` by matrix `b`, producing `a` * `b`.
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <param name="transpose_a">If `True`, `a` is transposed before multiplication.</param>
        /// <param name="transpose_b">If `True`, `b` is transposed before multiplication.</param>
        /// <param name="adjoint_a">If `True`, `a` is conjugated and transposed before multiplication.</param>
        /// <param name="adjoint_b">If `True`, `b` is conjugated and transposed before multiplication.</param>
        /// <param name="a_is_sparse">If `True`, `a` is treated as a sparse matrix.</param>
        /// <param name="b_is_sparse">If `True`, `b` is treated as a sparse matrix.</param>
        /// <param name="name">Name for the operation (optional).</param>
        /// <returns>
        /// A `Tensor` of the same type as `a` and `b` where each inner-most matrix is
        /// the product of the corresponding matrices in `a` and `b`, e.g. if all
        /// transpose or adjoint attributes are `False`:
        /// </returns>
        public static Tensor matmul(Tensor a, Tensor b,
            bool transpose_a = false, bool transpose_b = false,
            bool adjoint_a = false, bool adjoint_b = false,
            bool a_is_sparse = false, bool b_is_sparse = false,
            string name = null)
        {
            Tensor result = null;

            tf_with(ops.name_scope(name, "MatMul", new Tensor[] { a, b }), scope =>
            {
                name = scope;

                if (transpose_a && adjoint_a)
                    throw new ValueError("Only one of transpose_a and adjoint_a can be True.");
                if (transpose_b && adjoint_b)
                    throw new ValueError("Only one of transpose_b and adjoint_b can be True.");

                result = gen_math_ops.mat_mul(a, b, transpose_a, transpose_b, name);
            });

            return result;
        }

        public static Tensor batch_matmul(Tensor x, Tensor y,
            bool adj_x = false, bool adj_y = false,
            string name = null)
            => tf_with(ops.name_scope(name, "MatMul", new Tensor[] { x, y }), scope =>
            {
                name = scope;

                x = ops.convert_to_tensor(x, name: "a");
                y = ops.convert_to_tensor(y, name: "b");

                return tf.Context.ExecuteOp("BatchMatMul", name, new ExecuteOpArgs(x, y)
                    .SetAttributes(new { adj_x, adj_y }));
            });

        public static Tensor bincount(Tensor arr, Tensor weights = null,
             Tensor minlength = null,
             Tensor maxlength = null,
             TF_DataType dtype = TF_DataType.TF_INT32,
             string name = null,
             TensorShape axis = null,
             bool binary_output = false)
            => tf_with(ops.name_scope(name, "bincount"), scope =>
            {
                name = scope;
                if(!binary_output && axis == null)
                {
                    var array_is_nonempty = math_ops.reduce_prod(array_ops.shape(arr)) > 0;
                    var output_size = math_ops.cast(array_is_nonempty, dtypes.int32) * (math_ops.reduce_max(arr) + 1);
                    if (minlength != null)
                        output_size = math_ops.maximum(minlength, output_size);
                    if (maxlength != null)
                        output_size = math_ops.minimum(maxlength, output_size);
                    var weights = constant_op.constant(new long[0], dtype: dtype);
                    return tf.Context.ExecuteOp("Bincount", name, new ExecuteOpArgs(arr, output_size, weights));
                }

                throw new NotImplementedException("");
            });

        /// <summary>
        /// Returns the complex conjugate of a complex number.
        /// </summary>
        /// <param name="x">`Tensor` to conjugate.  Must have numeric or variant type.</param>
        /// <param name="name">A name for the operation (optional).</param>
        /// <returns>A `Tensor` that is the conjugate of `x` (with the same type).</returns>
        public static Tensor conj(Tensor x, string name = null)
        {
            var dt = x.dtype;
            if (dt.is_floating() || dt.is_integer())
                return x;

            return tf_with(ops.name_scope(name, "Conj", new List<Tensor> { x }), scope =>
            {

                return x;
            });
        }

        public static Tensor tanh(Tensor x, string name = null)
            => gen_math_ops.tanh(x, name);

        public static Tensor tensordot(Tensor x, Tensor y, int[] axes, string name = null)
        {
            Tensor _tensordot_reshape(Tensor a, int[] axes, bool flipped = false)
            {
                if (a.TensorShape.is_fully_defined() && isinstance(axes, (typeof(List<object>), typeof(Tuple))))
                {
                    var shape_a = a.TensorShape.as_list();

                    // axes
                    int iter = 0;
                    foreach (int i in axes)
                    {
                        if (i >= 0)
                            axes[0 + iter] = i;
                        else
                            axes[0 + iter] = i + len(shape_a);
                        iter++;
                    }

                    // free
                    int[] free = { };
                    iter = 0;
                    foreach (int i in Enumerable.Range(0, len(axes)))
                        if (!Array.Exists(axes, i => i == i))
                            free[free.Length] = i;

                    // free_dims
                    int[] free_dims = { };
                    foreach (int i in free)
                        free_dims[free_dims.Length] = shape_a[i];

                    int prod_free = (int)np.prod(free_dims);

                    // prod_axes
                    int[] prod_axes_pre = { };
                    foreach (int i in axes)
                        prod_axes_pre[prod_axes_pre.Length] = shape_a[i];
                    int prod_axes = (int)np.prod(prod_axes_pre);

                    // perm
                    Tensor perm;
                    if (flipped)
                        perm = ops.convert_to_tensor(list(free)) + ops.convert_to_tensor(free);
                    else
                        perm = ops.convert_to_tensor(list(free)) + ops.convert_to_tensor(free)
                                                                 + ops.convert_to_tensor(list(axes));

                    // new_shape
                    TensorShape new_shape;
                    if (flipped)
                        new_shape = new TensorShape(new int[] { prod_axes, prod_free });
                    else
                        new_shape = new TensorShape(new int[] { prod_free, prod_axes });
                }

                throw new NotImplementedException("_tensordot_reshape");
            }

            throw new NotImplementedException("tensordot");
        }

        public static Tensor truediv(Tensor x, Tensor y, string name = null)
            => _truediv_python3(x, y, name);

        public static Tensor _truediv_python3(Tensor x, Tensor y, string name = null)
        {
            return tf_with(ops.name_scope(name, "truediv", new { x, y }), scope =>
            {
                name = scope;
                var x_dtype = x.dtype.as_base_dtype();
                var y_dtype = y.dtype.as_base_dtype();

                if (x_dtype != y_dtype)
                    throw new TypeError($"x and y must have the same dtype, got {x_dtype} != {y_dtype}");

                var dtype = x_dtype switch
                {
                    TF_DataType.TF_UINT8 => TF_DataType.TF_FLOAT,
                    TF_DataType.TF_INT8 => TF_DataType.TF_FLOAT,
                    TF_DataType.TF_INT16 => TF_DataType.TF_FLOAT,
                    TF_DataType.TF_UINT16 => TF_DataType.TF_FLOAT,
                    TF_DataType.TF_INT32 => TF_DataType.TF_DOUBLE,
                    TF_DataType.TF_INT64 => TF_DataType.TF_DOUBLE,
                    _ => x_dtype
                };
                x = cast(x, dtype);
                y = cast(y, dtype);

                return gen_math_ops.real_div(x, y, name: name);
            });
        }
    }
}
