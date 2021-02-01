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
using Tensorflow.Contexts;
using Tensorflow.Eager;
using Tensorflow.Framework;
using static Tensorflow.Binding;

namespace Tensorflow
{
    public class array_ops
    {
        public static Tensor placeholder_with_default<T>(T input, int[] shape, string name = null)
            => gen_array_ops.placeholder_with_default(input, shape, name);

        /// <summary>
        ///    An identity op that triggers an error if a gradient is requested.
        /// </summary>
        /// <param name="input">
        ///    any tensor.
        /// </param>
        /// <param name="name">
        /// If specified, the created operation in the graph will be this one, otherwise it will be named 'PreventGradient'.
        /// </param>
        /// <param name="message">
        ///    Will be printed in the error when anyone tries to differentiate
        ///    this operation.
        /// </param>
        /// <returns>
        ///    the same input tensor.
        ///    The Operation can be fetched from the resulting Tensor, by fetching the Operation property from the result.
        /// </returns>
        /// <remarks>
        ///    When executed in a graph, this op outputs its input tensor as-is.
        ///    
        ///    When building ops to compute gradients, the TensorFlow gradient system
        ///    will return an error when trying to lookup the gradient of this op,
        ///    because no gradient must ever be registered for this function.  This
        ///    op exists to prevent subtle bugs from silently returning unimplemented
        ///    gradients in some corner cases.
        /// </remarks>
        public static Tensor prevent_gradient(Tensor input, string message = "", string name = null)
        {
            if (tf.executing_eagerly())
            {
                var results = tf.Runner.TFE_FastPathExecute(tf.Context, tf.Context.DeviceName,
                    "PreventGradient", name,
                    null,
                    input,
                    "message", message);
                return results[0];
            }

            var op = tf.OpDefLib._apply_op_helper("PreventGradient", name: name, args: new { input, message });
            return op.output;
        }

        internal static Tensor constant(object value,
            TF_DataType dtype = TF_DataType.DtInvalid,
            int[] shape = null,
            string name = "Const",
            bool verify_shape = false) => constant_op._constant_impl(value,
                dtype,
                shape,
                name,
                verify_shape: verify_shape,
                allow_broadcast: false);

        public static Tensor zeros(TensorShape shape, TF_DataType dtype = TF_DataType.TF_FLOAT, string name = null)
        {
            dtype = dtype.as_base_dtype();

            if (tf.executing_eagerly())
            {
                return tf_with(ops.name_scope(name, "zeros", shape), scope =>
                {
                    name = scope;
                    var shape_tensor = constant_op._tensor_shape_tensor_conversion_function(shape);
                    Tensor zeros = null;
                    switch (dtype)
                    {
                        case TF_DataType.TF_DOUBLE:
                            zeros = constant(0d);
                            break;
                        case TF_DataType.TF_FLOAT:
                            zeros = constant(0f);
                            break;
                        default:
                            zeros = constant(0);
                            break;
                    }
                    return fill(shape_tensor, zeros, name: name);
                });
            }
            else
            {
                return tf_with(ops.name_scope(name, "zeros", shape), scope =>
                {
                    name = scope;
                    switch (dtype)
                    {
                        case TF_DataType.TF_BOOL:
                            return _constant_if_small(false, shape, dtype, name);
                        case TF_DataType.TF_DOUBLE:
                            return _constant_if_small(0.0D, shape, dtype, name);
                        case TF_DataType.TF_FLOAT:
                            return _constant_if_small(0.0F, shape, dtype, name);
                        case TF_DataType.TF_INT64:
                            return _constant_if_small(0L, shape, dtype, name);
                        case TF_DataType.TF_INT32:
                            return _constant_if_small(0, shape, dtype, name);
                        case TF_DataType.TF_INT8:
                            return _constant_if_small<byte>(0, shape, dtype, name);
                        default:
                            throw new TypeError("can't find type for zeros");
                    }
                });
            }
        }

        public static Tensor boolean_mask<T1, T2>(T1 tensor, T2 mask, string name = "boolean_mask", int axis = 0)
        {
            return tf_with(ops.name_scope(name, values: new { tensor, mask }), delegate
            {
                var tensor_tensor = ops.convert_to_tensor(tensor, name: "tensor");
                var mask_tensor = ops.convert_to_tensor(mask, name: "mask");

                var shape_mask = mask_tensor.TensorShape;
                var ndims_mask = shape_mask.ndim;
                var shape_tensor = tensor_tensor.TensorShape;

                if (ndims_mask < 1)
                    throw new ValueError("mask cannot be scalar.");

                var leading_size = gen_math_ops.prod(shape(tensor_tensor)[$"{axis}:{axis + ndims_mask}"], new[] { 0 });
                var shape1 = concat(new[]
                {
                    shape(tensor_tensor)[$":{axis}"],
                    leading_size,
                    shape(tensor_tensor)[$"{axis + ndims_mask}:"]
                }, 0);
                tensor_tensor = reshape(tensor_tensor, shape1);
                var first_dim = shape_tensor.dims.Skip(axis).Take(ndims_mask).First();
                var s1 = tensor_shape.as_shape(shape_tensor.dims.Take(axis).ToArray());
                var s2 = s1.concatenate(new[] { first_dim }).concatenate(shape_tensor.dims.Skip(axis + ndims_mask).ToArray());
                tensor_tensor.set_shape(s2);

                mask_tensor = reshape(mask_tensor, new[] { -1 });
                return _apply_mask_1d(tensor_tensor, mask_tensor, axis);
            });
        }

        private static Tensor _apply_mask_1d(Tensor reshaped_tensor, Tensor mask, int axis = 0)
        {
            var indices = squeeze(where(mask), axis: new[] { 1 });
            return gather(reshaped_tensor, indices, axis: axis);
        }

        public static Tensor zeros(Tensor shape, TF_DataType dtype = TF_DataType.TF_FLOAT, string name = null)
        {
            dtype = dtype.as_base_dtype();
            return tf_with(ops.name_scope(name, "zeros", shape), scope =>
            {
                name = scope;
                switch (dtype)
                {
                    case TF_DataType.TF_BOOL:
                        return gen_array_ops.fill(shape, tf.constant(false, dtype: dtype), name: name);
                    case TF_DataType.TF_DOUBLE:
                        return gen_array_ops.fill(shape, tf.constant(0.0D, dtype: dtype), name: name);
                    case TF_DataType.TF_FLOAT:
                        return gen_array_ops.fill(shape, tf.constant(0.0F, dtype: dtype), name: name);
                    case TF_DataType.TF_INT32:
                        return gen_array_ops.fill(shape, tf.constant(0, dtype: dtype), name: name);
                    default:
                        throw new TypeError("can't find type for zeros");
                }

            });
        }

        private static Tensor _constant_if_small(int value, Tensor shape)
        {
            return shape < 1000;
        }

        private static Tensor _constant_if_small<T>(T value, TensorShape shape, TF_DataType dtype, string name)
        {
            Tensor shape_t = null;
            if (shape.size < 1000)
            {
                return constant_op.constant(value, shape: shape, dtype: dtype, name: name);
            }
            else
            {
                shape_t = constant_op._tensor_shape_tensor_conversion_function(shape);
                var c = constant_op.constant(0, dtype: dtype);
                return gen_array_ops.fill(shape_t, c, name: name);
            }
        }

        public static Tensor _autopacking_conversion_function(IEnumerable<object> v, TF_DataType dtype = TF_DataType.DtInvalid, string name = null, bool as_ref = false)
        {
            var inferred_dtype = _get_dtype_from_nested_lists(v);
            if (dtype == TF_DataType.DtInvalid)
                dtype = inferred_dtype;

            return _autopacking_helper(v, dtype, name == null ? "packed" : name);
        }

        private static TF_DataType _get_dtype_from_nested_lists(IEnumerable<object> list_or_tuple)
        {
            TF_DataType dtype = TF_DataType.DtInvalid;

            foreach (var obj in list_or_tuple)
            {
                switch (obj)
                {
                    case Tensor t:
                        dtype = t.dtype.as_base_dtype();
                        break;
                }

                if (dtype != TF_DataType.DtInvalid)
                    break;
            }

            return dtype;
        }

        /// <summary>
        /// Converts the given list or tuple to a tensor by packing.
        /// </summary>
        /// <param name="list_or_tuple">A (possibly nested) list or tuple containing a tensor.</param>
        /// <param name="dtype"></param>
        /// <param name="name"></param>
        /// <returns>A `tf.Tensor` with value equivalent to `list_or_tuple`.</returns>
        public static Tensor _autopacking_helper(IEnumerable<object> list_or_tuple, TF_DataType dtype, string name)
        {
            var must_pack = false;
            var converted_elems = new List<object>();

            bool switch_to_graph = tf.Context.switched_to_graph(list_or_tuple.ToArray());

            var result = tf_with(ops.name_scope(name), scope =>
            {
                foreach (var (i, elem) in enumerate(list_or_tuple))
                {
                    converted_elems.Add(elem);
                    must_pack = true;
                }

                if (must_pack)
                {
                    var elems_as_tensors = new List<Tensor>();
                    foreach (var (i, elem) in enumerate(converted_elems))
                    {
                        if (elem is EagerTensor eager_tensor)
                        {
                            if (switch_to_graph)
                                elems_as_tensors.Add(constant_op.constant(eager_tensor.numpy(), dtype: dtype, name: i.ToString()));
                            else
                                elems_as_tensors.Add(eager_tensor);
                        }
                        else if (elem is Tensor tensor)
                        {
                            elems_as_tensors.Add(tensor);
                        }
                        else
                        {
                            var elem_tensor = constant_op.constant(elem, dtype: dtype, name: i.ToString());
                            elems_as_tensors.Add(elem_tensor);
                        }
                    }

                    return gen_array_ops.pack(elems_as_tensors.ToArray(), name: scope);
                }
                else
                {
                    return tf.constant(np.array(new float[0]));
                }
            });

            if (switch_to_graph)
                tf.Context.restore_mode();

            return result;
        }

        public static Tensor expand_dims(Tensor input, int axis = -1, string name = null, int dim = -1)
            => expand_dims_v2(input, axis, name);

        private static Tensor expand_dims_v2(Tensor input, int axis, string name = null)
            => gen_array_ops.expand_dims(input, axis, name);

        /// <summary>
        /// Creates a tensor filled with a scalar value.
        /// This operation creates a tensor of shape `dims` and fills it with `value`.
        /// </summary>
        /// <param name="dims">A 1-D sequence of non-negative numbers.</param>
        /// <param name="value">A value to fill the returned `tf.Tensor`.</param>
        /// <param name="name">Optional string. The name of the output `tf.Tensor`.</param>
        /// <returns>A `tf.Tensor` with shape `dims` and the same dtype as `value`.</returns>
        public static Tensor fill(Tensor dims, Tensor value, string name = null)
        {
            var result = gen_array_ops.fill(dims, value, name: name);
            // tensor_util.maybe_set_static_shape(result, dims)
            return result;
        }

        /// <summary>
        /// Returns the rank of a tensor.
        /// </summary>
        /// <param name="input"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public static Tensor rank(Tensor input, string name = null)
            => rank_internal(input, name, optimize: true);

        public static Tensor rank(Tensor[] inputs, string name = null)
        {
            return tf_with(ops.name_scope(name, "Rank", new { inputs }), scope =>
            {
                name = scope;
                var input_tensor = ops.convert_to_tensor(inputs);
                return constant_op.constant(input_tensor.NDims, dtype: tf.int32, name: name);
            });
        }

        public static Tensor rank_internal(Tensor input, string name = null, bool optimize = true)
        {
            return tf_with(ops.name_scope(name, "Rank", new List<Tensor> { input }), scope =>
            {
                name = scope;
                var input_tensor = ops.convert_to_tensor(input);
                var input_shape = tensor_util.to_shape(input_tensor.shape);
                if (optimize && input_shape.ndim > 0)
                    return constant_op.constant(input_shape.ndim, dtype: tf.int32, name: name);
                else
                    return gen_array_ops.rank(input, name);
            });
        }

        /// <summary>
        /// Creates a tensor with all elements set to 1.
        /// </summary>
        /// <param name="tensor"></param>
        /// <param name="dtype"></param>
        /// <param name="name"></param>
        /// <param name="optimize"></param>
        /// <returns></returns>
        public static Tensor ones_like(Tensor tensor, TF_DataType dtype = TF_DataType.DtInvalid, string name = null, bool optimize = true)
        {
            return tf_with(ops.name_scope(name, "ones_like", new Tensor[] { tensor }), scope =>
            {
                name = scope;
                tensor = ops.convert_to_tensor(tensor, name: "tensor");

                // is_fully_defined return unexpected value.
                if (optimize && tensor_util.to_shape(tensor.shape).is_fully_defined() && dtype != TF_DataType.TF_VARIANT)
                {

                }

                if (dtype != TF_DataType.DtInvalid && dtype != tensor.dtype && dtype != TF_DataType.TF_VARIANT)
                {
                    throw new NotImplementedException("ones_like");
                    // return ones(shape_internal(tensor, optimize: optimize), dtype: dtype, name: name);
                }
                else
                {
                    return gen_array_ops.ones_like(tensor, name: name);
                }
            });
        }

        public static Tensor reshape(Tensor tensor, Tensor shape, string name = null)
            => gen_array_ops.reshape(tensor, shape, name: name);

        public static Tensor reshape(Tensor tensor, TensorShape shape, string name = null)
            => gen_array_ops.reshape(tensor, shape, name: name);

        public static Tensor reshape(Tensor tensor, object[] shape, string name = null)
            => gen_array_ops.reshape(tensor, shape, name: name);

        private static Tensor ones_like_impl<T>(T tensor, TF_DataType dtype, string name, bool optimize = true)
        {
            return tf_with(ops.name_scope(name, "ones_like", new { tensor }), scope =>
            {
                name = scope;
                var tensor1 = ops.convert_to_tensor(tensor, name: "tensor");
                var ones_shape = shape_internal(tensor1, optimize: optimize);
                if (dtype == TF_DataType.DtInvalid)
                    dtype = tensor1.dtype;
                var ret = ones(ones_shape, dtype: dtype, name: name);
                return ret;
            });
        }

        public static Tensor ones(Tensor shape, TF_DataType dtype = TF_DataType.TF_FLOAT, string name = null)
        {
            return tf_with(ops.name_scope(name, "ones", new { shape }), scope =>
            {
                name = scope;
                var output = gen_array_ops.fill(shape, constant_op.constant(1.0f, dtype: dtype), name: name);
                return output;
            });
        }

        public static Tensor ones(Tensor[] shape, TF_DataType dtype = TF_DataType.TF_FLOAT, string name = null)
        {
            dtype = dtype.as_base_dtype();
            return tf_with(ops.name_scope(name, "ones", new { shape }), scope =>
            {
                name = scope;
                var output = _constant_if_small(1, shape[0]);
                var shape1 = ops.convert_to_tensor(shape, dtype: TF_DataType.TF_INT32);
                output = gen_array_ops.fill(shape1, constant_op.constant(1, dtype: dtype), name: name);
                return output;
            });
        }

        public static Tensor ones(TensorShape shape, TF_DataType dtype = TF_DataType.TF_FLOAT, string name = null)
            => tf_with(ops.name_scope(name, "ones", shape), scope =>
            {
                dtype = dtype.as_base_dtype();
                name = scope;

                Tensor ones = null;
                switch (dtype)
                {
                    case TF_DataType.TF_DOUBLE:
                        ones = constant(1.0d);
                        break;
                    case TF_DataType.TF_FLOAT:
                        ones = constant(1.0f);
                        break;
                    default:
                        ones = constant(1);
                        break;
                }

                if (shape.ndim == 0)
                    return ones;

                var shape_tensor = constant_op._tensor_shape_tensor_conversion_function(shape);
                return fill(shape_tensor, ones, name: name);
            });

        public static Tensor one_hot(Tensor indices, Tensor depth,
            Tensor on_value = null,
            Tensor off_value = null,
            TF_DataType dtype = TF_DataType.DtInvalid,
            int axis = -1,
            string name = null)
        {
            return tf_with(ops.name_scope(name, "one_hot", new { indices, depth, dtype }), scope =>
            {
                name = scope;
                var on_exists = false;
                var off_exists = false;
                var on_dtype = TF_DataType.DtInvalid;
                var off_dtype = TF_DataType.DtInvalid;

                if (dtype == TF_DataType.DtInvalid)
                    dtype = TF_DataType.TF_FLOAT;

                if (!on_exists)
                {
                    on_value = ops.convert_to_tensor(1, dtype, name: "on_value");
                    on_dtype = dtype;
                }

                if (!off_exists)
                {
                    off_value = ops.convert_to_tensor(0, dtype, name = "off_value");
                    off_dtype = dtype;
                }

                return gen_array_ops.one_hot(indices, depth,
                    on_value: on_value,
                    off_value: off_value,
                    axis: axis,
                    name: name);
            });
        }

        public static (Tensor, Tensor) unique(Tensor x, TF_DataType out_idx = TF_DataType.TF_INT32, string name = null)
            => gen_array_ops.unique(x, out_idx: out_idx, name: name);

        public static Tensor stack(Tensor[] values, int axis = 0, string name = "stack")
        {
            if (axis == 0)
            {
                return ops.convert_to_tensor(values, name: name);
            }

            var value_shape = ops.convert_to_tensor(values[0], name: name).TensorShape;

            return gen_array_ops.pack(values, axis: axis, name: name);
        }

        public static Tensor[] unstack(Tensor value, int? num = null, int axis = 0, string name = "unstack")
        {
            if (num == null)
            {
                value = ops.convert_to_tensor(value);
                var value_shape = value.TensorShape;
                num = value_shape.dims[axis];
            }

            return gen_array_ops.unpack(value, num: num.Value, axis: axis, name: name);
        }

        public static Tensor where(Tensor condition, object x = null, object y = null, string name = null)
        {
            if (x == null && y == null)
            {
                return tf_with(ops.name_scope(name, "Where", new { condition }), scope =>
                {
                    name = scope;
                    condition = ops.convert_to_tensor(condition, preferred_dtype: dtypes.@bool, name: "condition");
                    return gen_array_ops.where(condition: condition, name: name);
                });
            }
            else if (x != null && y != null)
            {
                return gen_array_ops.select(condition, x, y, name);
            }
            else
            {
                throw new ValueError("x and y must both be non-None or both be None.");
            }
        }


        public static Tensor where_v2(Tensor condition, object x = null, object y = null, string name = null)
        {
            if (x == null && y == null)
            {
                return tf_with(ops.name_scope(name, "Where", new { condition }), scope =>
                {
                    name = scope;
                    condition = ops.convert_to_tensor(condition, preferred_dtype: dtypes.@bool, name: "condition");
                    return gen_array_ops.where(condition: condition, name: name);
                });
            }
            else if (x != null && y != null)
            {
                return gen_array_ops.select_v2(condition, x, y, name);
            }
            else
            {
                throw new ValueError("x and y must both be non-None or both be None.");
            }
        }
        /// <summary>
        /// Returns the shape of a tensor.
        /// </summary>
        /// <param name="input">A `Tensor` or `SparseTensor`.</param>
        /// <param name="name">A name for the operation (optional).</param>
        /// <param name="out_type">
        /// (Optional) The specified output type of the operation
        /// (`int32` or `int64`). Defaults to `tf.int32`.
        /// </param>
        /// <returns>A `Tensor` of type `out_type`.</returns>
        public static Tensor shape(Tensor input, string name = null, TF_DataType out_type = TF_DataType.TF_INT32)
            => shape_internal(input, name, optimize: true, out_type: out_type);

        public static Tensor shape_v2(Tensor input, string name = null, TF_DataType out_type = TF_DataType.TF_INT32)
            => shape_internal(input, name, optimize: true, out_type: out_type);

        public static Tensor size(Tensor input, string name = null, bool optimize = true, TF_DataType out_type = TF_DataType.TF_INT32)
            => size_internal(input, name, optimize: optimize, out_type: out_type);

        public static Tensor shape_internal(Tensor input, string name = null, bool optimize = true, TF_DataType out_type = TF_DataType.TF_INT32)
        {
            return tf_with(ops.name_scope(name, "Shape", new { input }), scope =>
            {
                name = scope;

                if (!tf.Context.executing_eagerly())
                {
                    var input_shape = input.TensorShape;
                    if (optimize && input.NDims > -1 && input_shape.is_fully_defined())
                    {
                        var nd = np.array(input.shape).astype(out_type.as_numpy_dtype());
                        return constant_op.constant(nd, name: name);
                    }
                }

                return gen_array_ops.shape(input, name: name, out_type: out_type);
            });
        }

        private static Tensor size_internal(Tensor input, string name = null, bool optimize = true, TF_DataType out_type = TF_DataType.TF_INT32)
        {
            return tf_with(ops.name_scope(name, "Size", new { input }), scope =>
            {
                name = scope;

                var input_tensor = ops.convert_to_tensor(input);
                var input_shape = tensor_util.to_shape(input_tensor.shape);
                if (optimize)
                {
                    if (input_shape.is_fully_defined())
                    {
                        return constant_op.constant(input_shape.size, dtype: out_type, name: name);
                    }
                }

                return gen_array_ops.size(input, name: name, out_type: out_type);
            });
        }

        public static Tensor tile(Tensor input, Tensor multiples, string name = null)
        {
            throw new NotImplementedException("tile");
        }

        public static Tensor zeros_like(Tensor tensor, TF_DataType dtype = TF_DataType.DtInvalid, string name = null, bool optimize = true)
        {
            return tf_with(ops.name_scope(name, "zeros_like", new Tensor[] { tensor }), scope =>
            {
                name = scope;
                tensor = ops.convert_to_tensor(tensor, name: "tensor");

                // is_fully_defined return unexpected value.
                if (optimize && tensor_util.to_shape(tensor.shape).is_fully_defined() && dtype != TF_DataType.TF_VARIANT)
                {

                }

                if (dtype != TF_DataType.DtInvalid && dtype != tensor.dtype && dtype != TF_DataType.TF_VARIANT)
                {
                    throw new NotImplementedException("zeros_like");
                    // return zeros(shape_internal(tensor, optimize: optimize), dtype: dtype, name: name);
                }
                else
                {
                    return gen_array_ops.zeros_like(tensor, name: name);
                }
            });
        }

        /// <summary>
        ///   When building ops to compute gradients, this op prevents the contribution of
        ///   its inputs to be taken into account.Normally, the gradient generator adds ops
        ///   to a graph to compute the derivatives of a specified 'loss' by recursively
        ///   finding out inputs that contributed to its computation.If you insert this op
        ///   in the graph it inputs are masked from the gradient generator.  They are not
        ///   taken into account for computing gradients.
        /// </summary>
        /// <param name="input"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public static Tensor stop_gradient(Tensor input, string name = null)
            => gen_array_ops.stop_gradient(input, name);

        /// <summary>
        /// Extracts a strided slice of a tensor (generalized python array indexing).
        /// </summary>
        /// <param name="input_"></param>
        /// <param name="begin"></param>
        /// <param name="end"></param>
        /// <param name="strides"></param>
        /// <param name="begin_mask"></param>
        /// <param name="end_mask"></param>
        /// <param name="ellipsis_mask"></param>
        /// <param name="new_axis_mask"></param>
        /// <param name="shrink_axis_mask"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public static Tensor strided_slice(Tensor input_, Tensor begin, Tensor end,
            Tensor strides = null,
            int begin_mask = 0,
            int end_mask = 0,
            int ellipsis_mask = 0,
            int new_axis_mask = 0,
            int shrink_axis_mask = 0,
            string name = null)
        {
            var op = gen_array_ops.strided_slice(
                input: input_,
                begin: begin,
                end: end,
                strides: strides,
                begin_mask: begin_mask,
                end_mask: end_mask,
                ellipsis_mask: ellipsis_mask,
                new_axis_mask: new_axis_mask,
                shrink_axis_mask: shrink_axis_mask,
                name: name);

            string parent_name = name;

            return op;
        }

        /// <summary>
        /// Returns the gradient of `StridedSlice`.
        /// 
        /// Since `StridedSlice` cuts out pieces of its `input` which is size
        /// `shape`, its gradient will have the same shape (which is passed here
        /// as `shape`). The gradient will be zero in any element that the slice
        /// does not select.
        /// </summary>
        /// <param name="shape">Must be one of the following types: `int32`, `int64`.</param>
        /// <param name="begin">Must have the same type as `shape`.</param>
        /// <param name="end">Must have the same type as `shape`.</param>
        /// <param name="strides">Must have the same type as `shape`.</param>
        /// <param name="dy">A `Tensor`.</param>
        /// <param name="begin_mask">An optional `int`. Defaults to `0`.</param>
        /// <param name="end_mask">An optional `int`. Defaults to `0`.</param>
        /// <param name="ellipsis_mask">An optional `int`. Defaults to `0`.</param>
        /// <param name="new_axis_mask">An optional `int`. Defaults to `0`.</param>
        /// <param name="shrink_axis_mask">An optional `int`. Defaults to `0`.</param>
        /// <param name="name">A name for the operation (optional).</param>
        /// <returns>A `Tensor`. Has the same type as `dy`.</returns>
        public static Tensor strided_slice_grad(Tensor shape, Tensor begin, Tensor end, Tensor strides, Tensor dy,
            long begin_mask = 0, long end_mask = 0, long ellipsis_mask = 0, long new_axis_mask = 0,
            long shrink_axis_mask = 0, string name = null)
            => tf.Context.RunInAutoMode2(
                () => tf.OpDefLib._apply_op_helper("StridedSliceGrad", name, new
                {
                    shape,
                    begin,
                    end,
                    strides,
                    dy,
                    begin_mask,
                    end_mask,
                    ellipsis_mask,
                    new_axis_mask,
                    shrink_axis_mask
                }).output,
                () => tf.Runner.TFE_FastPathExecute(tf.Context, tf.Context.DeviceName,
                    "StridedSliceGrad", name,
                    null,
                    shape, begin, end, strides, dy,
                    "begin_mask", begin_mask,
                    "end_mask", end_mask,
                    "ellipsis_mask", ellipsis_mask,
                    "new_axis_mask", new_axis_mask,
                    "shrink_axis_mask", shrink_axis_mask).FirstOrDefault(),
                (op) =>
                {
                    var attrs = new object[]
                    {
                        "T", op.get_attr<TF_DataType>("T"),
                        "Index", op.get_attr<TF_DataType>("Index"),
                        "begin_mask", op.get_attr<long>("begin_mask"),
                        "end_mask", op.get_attr<long>("end_mask"),
                        "ellipsis_mask", op.get_attr<long>("ellipsis_mask"),
                        "new_axis_mask", op.get_attr<long>("new_axis_mask"),
                        "shrink_axis_mask", op.get_attr<long>("shrink_axis_mask")
                    };
                    tf.Runner.RecordGradient("StridedSliceGrad", op.inputs, attrs, op.outputs);
                },
                new Tensors(shape, begin, end, strides, dy));

        /// <summary>
        /// Removes dimensions of size 1 from the shape of a tensor.
        /// Given a tensor `input`, this operation returns a tensor of the same type with
        /// all dimensions of size 1 removed.If you don't want to remove all size 1
        /// dimensions, you can remove specific size 1 dimensions by specifying
        /// `axis`.
        /// </summary>
        /// <param name="input"> A `Tensor`. The `input` to squeeze.</param>
        /// <param name="axis"> An optional list of `ints`. Defaults to `[]`.
        /// If specified, only squeezes the dimensions listed.The dimension
        /// index starts at 0. It is an error to squeeze a dimension that is not 1.
        /// Must be in the range `[-rank(input), rank(input))`.</param>
        /// <param name="name"> A name for the operation (optional).</param>
        /// <param name="squeeze_dims" >Deprecated keyword argument that is now axis.</param>
        /// <returns>A `Tensor`. Has the same type as `input`.
        /// Contains the same data as `input`, but has one or more dimensions of
        /// size 1 removed.</returns>
        public static Tensor squeeze(Tensor input, int[] axis = null, string name = null, int[] squeeze_dims = null)
            => gen_array_ops.squeeze(input, axis, name);

        public static Tensor identity(Tensor input, string name = null)
            => gen_array_ops.identity(input, name);

        public static Tensor invert_permutation(Tensor x, string name = null)
            => gen_array_ops.invert_permutation(x, name: name);

        public static Tensor matrix_diag(Tensor diagonal,
                string name = "diag",
                int k = 0,
                int num_rows = -1,
                int num_cols = -1,
                float padding_value = 0,
                string align = "RIGHT_LEFT")
        {
            if (tf.Context.executing_eagerly())
            {
                var results = tf.Runner.TFE_FastPathExecute(tf.Context, tf.Context.DeviceName,
                    "MatrixDiagV3", name,
                    null,
                    diagonal, k, num_rows, num_cols, padding_value,
                    "align", align);
                return results[0];
            }

            throw new NotImplementedException("");
        }

        public static Tensor matrix_set_diag(Tensor input,
            Tensor diagonal,
            string name = "set_diag",
            int k = 0,
            string align = "RIGHT_LEFT")
        {
            if (tf.Context.executing_eagerly())
            {
                var results = tf.Runner.TFE_FastPathExecute(tf.Context, tf.Context.DeviceName,
                    "MatrixSetDiagV3", name,
                    null,
                    input, diagonal, k,
                    "align", align);
                return results[0];
            }

            throw new NotImplementedException("");
        }

        /// <summary>
        /// Computes the shape of a broadcast given symbolic shapes.
        /// When shape_x and shape_y are Tensors representing shapes(i.e.the result of
        /// calling tf.shape on another Tensor) this computes a Tensor which is the shape
        /// of the result of a broadcasting op applied in tensors of shapes shape_x and
        /// shape_y.
        /// For example, if shape_x is [1, 2, 3] and shape_y is [5, 1, 3], the result is a
        /// Tensor whose value is [5, 2, 3].
        /// This is useful when validating the result of a broadcasting operation when the
        /// tensors do not have statically known shapes.
        /// </summary>
        /// <param name="shape_x"> A rank 1 integer `Tensor`, representing the shape of x.</param>
        /// <param name="shape_y"> A rank 1 integer `Tensor`, representing the shape of y.</param>
        /// <returns> A rank 1 integer `Tensor` representing the broadcasted shape.</returns>
        public static Tensor broadcast_dynamic_shape(Tensor shape_x, Tensor shape_y)
            => gen_array_ops.broadcast_args(shape_x, shape_y);

        public static Tensor broadcast_static_shape(Tensor shape_x, Tensor shape_y)
            => Framework.common_shapes.broadcast_shape(shape_x, shape_y);

        /// <summary>
        /// Concatenates tensors along one dimension.
        /// </summary>
        /// <param name="values"></param>
        /// <param name="axis"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public static Tensor concat(Tensor[] values, int axis, string name = "concat")
        {
            if (values.Length == 1) // Degenerate case of one tensor.
            {
                return tf_with(ops.name_scope(name), scope =>
                {
                    var t = ops.convert_to_tensor(axis, name: "concat_dim", dtype: TF_DataType.TF_INT32);
                    return identity(values[0], name: scope);
                });
            }

            return gen_array_ops.concat_v2(values, axis, name: name);
        }

        public static Tensor concat(Tensor[] values, Tensor axis, string name = "concat")
        {
            return gen_array_ops.concat_v2(values, axis, name: name);
        }

        public static Tensor concat(object[] values, int axis, string name = "concat")
        {
            return gen_array_ops.concat_v2(values, axis, name: name);
        }

        public static Tensor gather<T1, T2>(T1 @params, T2 indices, string name = null, int axis = 0)
        {
            if (axis != 0)
                return gen_array_ops.gather_v2(@params, indices, axis, name: name);

            if (@params is ResourceVariable variable &&
                indices is Tensor indices_tensor)
                return variable.sparse_read(indices_tensor, name);

            return gen_array_ops.gather_v2(@params, indices, axis, name: name);
        }

        public static Tensor transpose<T1>(T1 a, TensorShape perm, string name = "transpose", bool conjugate = false)
        {
            return tf_with(ops.name_scope(name, "transpose", new { a }), scope =>
            {
                var a_tensor = ops.convert_to_tensor(a);
                if (perm == null)
                {
                    var rank = a_tensor.rank;
                    perm = range(0, rank).OrderByDescending(x => x).ToArray();
                }

                return gen_array_ops.transpose(a_tensor, perm, name: scope);
            });
        }

        public static Tensor transpose(Tensor a, Tensor perm, string name = "transpose", bool conjugate = false)
        {
            return tf_with(ops.name_scope(name, "transpose", new { a }), scope =>
            {
                return gen_array_ops.transpose(a, perm, name: scope);
            });
        }

        public static Tensor[] split(Tensor value, Tensor size_splits, int axis, int num = -1,
            string name = "split")
        {
            if (num == -1)
                num = size_splits.shape[0];

            return gen_array_ops.split_v(value, size_splits, axis, num, name: name);
        }

        public static Tensor[] split<T>(Tensor value, int num_split, T axis,
            string name = "split")
        {
            var size_splits = ops.convert_to_tensor(num_split);

            if (tf.Context.executing_eagerly())
            {
                return split_eager_fallback(axis, value, num_split: num_split, name: name, ctx: tf.Context);
            }

            var _op = tf.OpDefLib._apply_op_helper("Split", name, new { split_dim = axis, value, num_split });
            return _op.outputs;
        }

        private static Tensor[] split_eager_fallback<Ta, Tv>(Ta axis, Tv value, int num_split, string name, Context ctx = null)
        {
            var (_attr_T, input) = tf.Runner.ArgsToMatchingEager(ctx, args: new object[] { value });
            var axis_tensor = ops.convert_to_tensor(axis, dtype: TF_DataType.TF_INT32);
            var _inputs_flat = new List<Tensor> { axis_tensor };
            _inputs_flat.AddRange(input);
            var _attrs = new object[] { "num_split", num_split, "T", _attr_T };

            return tf.Runner.Execute(ctx, "Split", num_split, _inputs_flat.ToArray(), _attrs, name: name);
        }

        public static Tensor slice(Tensor input, Tensor[] begin, Tensor[] size, string name = null)
            => gen_array_ops.slice(input, begin, size, name: name);

        public static Tensor slice<Tb, Ts>(Tensor input, Tb begin, Ts size, string name = null)
            => gen_array_ops.slice(input, begin, size, name: name);

        public static Tensor slice(Tensor input, Tensor begin, Tensor size, string name = null)
            => tf.Context.RunInAutoMode2(
                () => tf.OpDefLib._apply_op_helper("Slice", name, new
                {
                    input,
                    begin,
                    size
                }).output,
                () => tf.Runner.TFE_FastPathExecute(tf.Context, tf.Context.DeviceName,
                    "Slice", name,
                    null,
                    input, begin, size).FirstOrDefault(),
                (op) =>
                {
                    var attrs = new object[]
                    {
                        "T", op.get_attr<TF_DataType>("T"),
                        "Index", op.get_attr<int>("Index")
                    };
                    tf.Runner.RecordGradient("Slice", op.inputs, attrs, op.outputs);
                },
                new Tensors(input, begin, size));

        public static Tensor stack(object values, int axis = 0, string name = "stack")
        {
            if (axis == 0)
                // If the input is a constant list, it can be converted to a constant op
                return ops.convert_to_tensor(values, name: name);

            throw new NotImplementedException("array_ops.stack");
        }

        public static Tensor pad(Tensor tensor, Tensor paddings, string mode = "CONSTANT", string name = null, int constant_values = 0)
        {
            Tensor result = null;
            mode = mode.ToUpper();
            if (mode == "CONSTANT")
            {
                if (constant_values != 0)
                    throw new NotImplementedException("gen_array_ops.pad_v2");
                else
                    result = gen_array_ops.pad(tensor, paddings, name: name);
            }

            // Restore shape information where possible.
            if (!tf.Context.executing_eagerly())
            {
                var paddings_constant = tensor_util.constant_value(paddings);
                var input_shape = result.op.inputs[0].TensorShape;
                if (input_shape.ndim > -1 &&
                    !result.TensorShape.is_fully_defined() &&
                    !(paddings_constant is null))
                {
                    var new_shape = new List<int>();
                    foreach ((NDArray padding, int dim) in zip(paddings_constant.GetNDArrays(), np.array(input_shape.dims).GetNDArrays()))
                    {
                        if (padding is null || dim == -1 || padding.GetData<int>().Contains(-1))
                            new_shape.Add(-1);
                        else
                            new_shape.Add(np.sum(padding) + dim);
                    }
                    result.set_shape(new_shape.ToArray());
                }
            }

            return result;
        }

        public static Tensor placeholder(TF_DataType dtype, TensorShape shape = null, string name = null)
        {
            if (tf.Context.executing_eagerly())
                throw new RuntimeError("tf.placeholder() is not compatible with eager execution.");

            var _op = tf.OpDefLib._apply_op_helper("Placeholder", name: name, args: new { dtype, shape });
            return _op.output;
        }
    }
}
