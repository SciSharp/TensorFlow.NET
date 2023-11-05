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

using Tensorflow.NumPy;
using System;
using System.Collections.Generic;
using System.Linq;
using Tensorflow.Contexts;
using Tensorflow.Eager;
using Tensorflow.Framework;
using static Tensorflow.Binding;
using System.Diagnostics;

namespace Tensorflow
{
    public class array_ops
    {
        public static Tensor placeholder_with_default(Tensor input, int[] shape, string name = null)
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
            => tf.Context.ExecuteOp("PreventGradient", name, new ExecuteOpArgs(input)
                .SetAttributes(new { message }));

        internal static Tensor constant(object value,
            TF_DataType dtype = TF_DataType.DtInvalid,
            int[] shape = null,
            string name = "Const",
            bool verify_shape = false) => constant_op.constant(value,
                dtype: dtype,
                shape: shape,
                name: name,
                verify_shape: verify_shape,
                allow_broadcast: false);

        public static Tensor zeros(Shape shape, TF_DataType dtype = TF_DataType.TF_FLOAT, string name = null)
        {
            dtype = dtype.as_base_dtype();

            if (tf.executing_eagerly())
            {
                return tf_with(ops.name_scope(name, "zeros", shape), scope =>
                {
                    name = scope;
                    // var shape_tensor = constant_op._tensor_shape_tensor_conversion_function(shape);
                    Tensor zeros = dtype switch
                    {
                        TF_DataType.TF_BOOL => constant(false),
                        TF_DataType.TF_DOUBLE => constant(0d),
                        TF_DataType.TF_FLOAT => constant(0f),
                        TF_DataType.TF_INT64 => constant(0L),
                        TF_DataType.TF_UINT64 => constant((ulong)0),
                        TF_DataType.TF_INT32 => constant(0),
                        TF_DataType.TF_UINT32 => constant((uint)0),
                        TF_DataType.TF_INT8 => constant((sbyte)0),
                        TF_DataType.TF_UINT8 => constant((byte)0),
                        _ => constant(0)
                    };
                    return fill(shape, zeros, name: name);
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
                        case TF_DataType.TF_UINT64:
                            return _constant_if_small<ulong>(0, shape, dtype, name);
                        case TF_DataType.TF_INT32:
                            return _constant_if_small(0, shape, dtype, name);
                        case TF_DataType.TF_UINT32:
                            return _constant_if_small<uint>(0, shape, dtype, name);
                        case TF_DataType.TF_INT8:
                            return _constant_if_small<sbyte>(0, shape, dtype, name);
                        case TF_DataType.TF_UINT8:
                            return _constant_if_small<byte>(0, shape, dtype, name);
                        default:
                            throw new TypeError("can't find type for zeros");
                    }
                });
            }
        }

        public static Tensor zeros(Tensors shape, TF_DataType dtype = TF_DataType.TF_FLOAT, string name = null)
        {
            dtype = dtype.as_base_dtype();
            Tensor shapeTensor;
            if(shape.Length > 1)
            {
                shapeTensor = ops.convert_to_tensor(shape, dtypes.int32);
                if (shapeTensor.ndim > 1)
                {
                    shapeTensor = array_ops.reshape(shapeTensor, new Shape(-1));
                }
            }
            else
            {
                shapeTensor = shape[0];
            }
            var output = fill(shapeTensor, array_ops.constant(0, dtype), name);
            Debug.Assert(output.dtype.as_base_dtype() == dtype);
            return output;
        }

        public static Tensor boolean_mask<T1, T2>(T1 tensor, T2 mask, string name = "boolean_mask", int axis = 0)
        {
            return tf_with(ops.name_scope(name, values: new { tensor, mask }), delegate
            {
                var tensor_tensor = ops.convert_to_tensor(tensor, name: "tensor");
                var mask_tensor = ops.convert_to_tensor(mask, name: "mask");

                var shape_mask = mask_tensor.shape;
                var ndims_mask = shape_mask.ndim;
                var shape_tensor = tensor_tensor.shape;

                if (ndims_mask < 1)
                    throw new ValueError("mask cannot be scalar.");

                var leading_size = gen_math_ops.prod(shape(tensor_tensor)[$"{axis}:{axis + ndims_mask}"], ops.convert_to_tensor(new[] { 0 }));
                if (leading_size.rank == 0)
                {
                    leading_size = expand_dims(leading_size, 0);
                }

                var shape1 = concat(new[]
                {
                    shape(tensor_tensor)[$":{axis}"],
                    leading_size,
                    shape(tensor_tensor)[$"{axis + ndims_mask}:"]
                }, 0);
                tensor_tensor = reshape(tensor_tensor, shape1);
                var first_dim = shape_tensor.dims.Skip(axis).Take(ndims_mask).First();
                var s1 = new Shape(shape_tensor.dims.Take(axis).ToArray());
                var s2 = s1.concatenate(new[] { first_dim }).concatenate(shape_tensor.dims.Skip(axis + ndims_mask).ToArray());
                tensor_tensor.shape = s2;

                mask_tensor = reshape(mask_tensor, new[] { -1 });
                return _apply_mask_1d(tensor_tensor, mask_tensor, axis);
            });
        }

        private static Tensor _apply_mask_1d(Tensor reshaped_tensor, Tensor mask, int axis = 0)
        {
            var indices = squeeze(where_v2(mask), axis: new[] { 1 });
            return gather(reshaped_tensor, indices, axis: ops.convert_to_tensor(axis));
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
            if (shape.dtype == TF_DataType.TF_INT64)
                return shape < 1000L;
            else
                return shape < 1000;
        }

        private static Tensor _constant_if_small<T>(T value, Shape shape, TF_DataType dtype, string name)
        {
            if (shape.size < 1000)
            {
                return constant_op.constant(value, shape: shape, dtype: dtype, name: name);
            }
            else
            {
                var shape_t = constant_op._tensor_shape_tensor_conversion_function(shape);
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

        private static TF_DataType _get_dtype_from_nested_lists<T>(IEnumerable<T> list_or_tuple)
        {
            TF_DataType dtype = TF_DataType.DtInvalid;

            foreach (var obj in list_or_tuple)
            {
                switch (obj)
                {
                    case Tensor t:
                        dtype = t.dtype.as_base_dtype();
                        break;
                    case int t:
                        dtype = TF_DataType.TF_INT32;
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
                        else if (elem is KerasTensor kt)
                        {
                            elems_as_tensors.Add(kt);
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

        public static Tensor expand_dims(Tensor input, int axis = -1, string name = null)
            => gen_array_ops.expand_dims(input, ops.convert_to_tensor(axis), name);

        /// <summary>
        /// Creates a tensor filled with a scalar value.
        /// This operation creates a tensor of shape `dims` and fills it with `value`.
        /// </summary>
        /// <param name="dims">A 1-D sequence of non-negative numbers.</param>
        /// <param name="value">A value to fill the returned `tf.Tensor`.</param>
        /// <param name="name">Optional string. The name of the output `tf.Tensor`.</param>
        /// <returns>A `tf.Tensor` with shape `dims` and the same dtype as `value`.</returns>
        public static Tensor fill<T>(Shape dims, T value, string name = null)
            => gen_array_ops.fill(dims, ops.convert_to_tensor(value), name: name);

        public static Tensor fill<T>(Tensor dims, T value, string name = null)
            => gen_array_ops.fill(dims, ops.convert_to_tensor(value), name: name);

        /// <summary>
        /// Returns the rank of a tensor.
        /// </summary>
        /// <param name="input"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public static Tensor rank(Tensor input, string name = null)
            => rank_internal(input, name, optimize: true);

        public static Tensor rank_internal(Tensor input, string name = null, bool optimize = true)
        {
            return tf_with(ops.name_scope(name, "Rank", new List<Tensor> { input }), scope =>
            {
                name = scope;
                var input_shape = input.shape;
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
                if (optimize && tensor.shape.IsFullyDefined && dtype != TF_DataType.TF_VARIANT)
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

        public static Tensor reshape(Tensor tensor, Shape shape, string name = null)
            => gen_array_ops.reshape(tensor, shape, name: name);

        public static Tensor reshape(Tensor tensor, object[] shape, string name = null)
        {
            var dims = shape_utils.from_object_array(shape);
            return gen_array_ops.reshape(tensor, dims, name: name);
        }

        public static Tensor reverse(Tensor tensor, Tensor axis, string name = null)
            => tf.Context.ExecuteOp("ReverseV2", name, new ExecuteOpArgs(tensor, axis)
            {
                GetGradientAttrs = (op) => new
                {
                    T = op.get_attr<TF_DataType>("T"),
                    Tidx = op.get_attr<TF_DataType>("Tidx")
                }
            });

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
                if (shape._shape_tuple().Length == 0)
                {
                    shape = reshape(shape, new Shape(-1));
                }
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

        public static Tensor ones(Shape shape, TF_DataType dtype = TF_DataType.TF_FLOAT, string name = null)
            => tf_with(ops.name_scope(name, "ones", shape), scope =>
            {
                dtype = dtype.as_base_dtype();
                name = scope;

                Tensor ones = dtype switch
                {
                    TF_DataType.TF_DOUBLE => constant(1.0d),
                    TF_DataType.TF_FLOAT => constant(1.0f),
                    _ => constant(1, dtype)
                };

                if (shape.ndim == 0)
                    return ones;

                // var shape_tensor = constant_op._tensor_shape_tensor_conversion_function(shape);
                return fill(shape, ones, name: name);
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
        {
            var res = gen_array_ops.unique(x, out_idx: out_idx, name: name);
            Debug.Assert(res.Length == 2);
            return (res[0], res[1]);
        }

        public static Tensor stack(Tensor[] values, int axis = 0, string name = "stack")
        {
            if (axis == 0)
            {
                return ops.convert_to_tensor(values, name: name);
            }

            return gen_array_ops.pack(values, axis: axis, name: name);
        }

        public static Tensor[] unstack(Tensor value, int? num = null, int axis = 0, string name = "unstack")
        {
            num = num ?? value.shape.as_int_list()[axis];
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
                    return gen_array_ops.where(condition, name: name);
                });
            }
            else if (x != null && y != null)
            {
                return gen_math_ops.select(condition, ops.convert_to_tensor(x), ops.convert_to_tensor(y), name);
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
                    return gen_array_ops.where(condition, name: name);
                });
            }
            else if (x != null && y != null)
            {
                return gen_math_ops.select_v2(condition, ops.convert_to_tensor(x), ops.convert_to_tensor(y), name);
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

        public static Tensor size<T>(T input, string name = null, bool optimize = true, TF_DataType out_type = TF_DataType.TF_INT32)
            => size_internal(input, name, optimize: optimize, out_type: out_type);

        public static Tensor shape_internal(Tensor input, string name = null, bool optimize = true, TF_DataType out_type = TF_DataType.TF_INT32)
        {
            return tf_with(ops.name_scope(name, "Shape", new { input }), scope =>
            {
                name = scope;

                if (!tf.Context.executing_eagerly())
                {
                    var input_shape = input.shape;
                    if (optimize && input.ndim > -1 && input_shape.IsFullyDefined)
                    {
                        if(out_type == TF_DataType.TF_INT32)
                            return constant_op.constant(input.shape.as_int_list(), name: name);
                        else
                            return constant_op.constant(input.shape.dims, name: name);
                    }
                }

                return tf.Context.ExecuteOp("Shape", name, new ExecuteOpArgs(input)
                {
                    GetGradientAttrs = (op) => new
                    {
                        T = op.get_attr<TF_DataType>("T"),
                        out_type = op.get_attr<TF_DataType>("out_type")
                    }
                }.SetAttributes(new
                {
                    out_type
                })).First();
            });
        }

        private static Tensor size_internal<T>(T input, string name = null, bool optimize = true, TF_DataType out_type = TF_DataType.TF_INT32)
        {
            return tf_with(ops.name_scope(name, "Size", new { input }), scope =>
            {
                name = scope;

                var input_tensor = ops.convert_to_tensor(input);
                var input_shape = input_tensor.shape;
                if (optimize)
                {
                    if (input_shape.IsFullyDefined)
                    {
                        return constant_op.constant(input_shape.size, dtype: out_type, name: name);
                    }
                }

                return gen_array_ops.size(input_tensor, name: name, out_type: out_type);
            });
        }

        public static Tensor tile(Tensor input, Tensor multiples, string name = null)
            => tf.Context.ExecuteOp("Tile", name, new ExecuteOpArgs(input, multiples)
            {
                GetGradientAttrs = (op) => new
                {
                    T = op.get_attr<TF_DataType>("T"),
                    Tmultiples = op.get_attr<TF_DataType>("Tmultiples")
                }
            });

        /*public static Tensor tile(Tensor input, Shape multiples, string name = null)
        {
            return tf.Context.ExecuteOp("Tile", name, new ExecuteOpArgs(input, multiples)
            {
                GetGradientAttrs = (op) => new
                {
                    T = op.get_attr<TF_DataType>("T"),
                    Tmultiples = op.get_attr<TF_DataType>("Tmultiples")
                }
            });
        }*/

        public static Tensor zeros_like(Tensor tensor, TF_DataType dtype = TF_DataType.DtInvalid, string name = null, bool optimize = true)
        {
            return tf_with(ops.name_scope(name, "zeros_like", new Tensor[] { tensor }), scope =>
            {
                name = scope;
                tensor = ops.convert_to_tensor(tensor, name: "tensor");

                // is_fully_defined return unexpected value.
                if (optimize && tensor.shape.IsFullyDefined && dtype != TF_DataType.TF_VARIANT)
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
        {
            var tape = tf.GradientTape().stop_recording();
            var result = gen_array_ops.stop_gradient(input, name);
            tape.StartRecord();
            return result;
        }

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
                => tf.Context.ExecuteOp("StridedSlice", name, new ExecuteOpArgs(input_, begin, end, strides)
                {
                    GetGradientAttrs = (op) => new
                    {
                        T = op.get_attr<TF_DataType>("T"),
                        Index = op.get_attr<TF_DataType>("Index"),
                        begin_mask = op.get_attr<long>("begin_mask"),
                        end_mask = op.get_attr<long>("end_mask"),
                        ellipsis_mask = op.get_attr<long>("ellipsis_mask"),
                        new_axis_mask = op.get_attr<long>("new_axis_mask"),
                        shrink_axis_mask = op.get_attr<long>("shrink_axis_mask")
                    }
                }.SetAttributes(new
                {
                    begin_mask,
                    end_mask,
                    ellipsis_mask,
                    new_axis_mask,
                    shrink_axis_mask
                }));

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
            => tf.Context.ExecuteOp("StridedSliceGrad", name,
                new ExecuteOpArgs(shape, begin, end, strides, dy)
                {
                    GetGradientAttrs = (op) => new
                    {
                        T = op.get_attr<TF_DataType>("T"),
                        Index = op.get_attr<TF_DataType>("Index"),
                        begin_mask = op.get_attr<long>("begin_mask"),
                        end_mask = op.get_attr<long>("end_mask"),
                        ellipsis_mask = op.get_attr<long>("ellipsis_mask"),
                        new_axis_mask = op.get_attr<long>("new_axis_mask"),
                        shrink_axis_mask = op.get_attr<long>("shrink_axis_mask")
                    }
                }.SetAttributes(new
                {
                    begin_mask,
                    end_mask,
                    ellipsis_mask,
                    new_axis_mask,
                    shrink_axis_mask
                }));

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
        public static Tensor squeeze(Tensor input, Axis axis = null, string name = null)
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
                float padding_value = 0f,
                string align = "RIGHT_LEFT")
            => tf.Context.ExecuteOp("MatrixDiagV3", name, 
                new ExecuteOpArgs(diagonal, k, num_rows, num_cols, ops.convert_to_tensor(padding_value, dtype: diagonal.dtype))
                    .SetAttributes(new { align }));

        public static Tensor matrix_set_diag(Tensor input,
            Tensor diagonal,
            string name = "set_diag",
            int k = 0,
            string align = "RIGHT_LEFT")
                => tf.Context.ExecuteOp("MatrixSetDiagV3", name, new ExecuteOpArgs(input, diagonal, k)
                    .SetAttributes(new { align }));

        public static Tensor[] meshgrid<T>(T[] array, bool copy = true, bool sparse = false, string indexing = "xy")
        {
            return tf_with(ops.name_scope(null, "meshgrid", array), scope =>
            {
                var ndim = array.Length;
                var s0 = range(ndim).Select(x => 1).ToArray();

                var output = new List<Tensor>();
                foreach (var (i, x) in enumerate(array))
                {
                    var shape = s0.Take(i).ToArray().concat(new[] { -1 }).concat(s0.Skip(i + 1).ToArray());
                    output.add(reshape(stack(x), shape));
                }

                // Create parameters for broadcasting each tensor to the full size
                var shapes = array.Select(x => size(x)).ToArray();
                var output_dtype = _get_dtype_from_nested_lists(array).as_base_dtype();
                if (indexing == "xy" && ndim > 1)
                {
                    output[0] = reshape(output[0], new[] { 1, -1 }.concat(range(ndim - 2).Select(x => 1).ToArray()));
                    output[1] = reshape(output[1], new[] { -1, 1 }.concat(range(ndim - 2).Select(x => 1).ToArray()));
                    (shapes[0], shapes[1]) = (shapes[1], shapes[0]);
                }

                if(sparse)
                    return output.ToArray();
                else
                {
                    var mult_fact = ones(shapes, output_dtype);
                    return output.Select(x => x * mult_fact).ToArray();
                }
            });
        }

        public static Tensor moveaxis(NDArray array, Axis source, Axis destination)
        {
            List<int> perm = null;
            source = source.axis.Select(x => x < 0 ? array.rank + x : x).ToArray();
            destination = destination.axis.Select(x => x < 0 ? array.rank + x : x).ToArray();

            if (array.shape.rank > -1)
            {
                perm = range(0, array.rank).Where(i => !source.axis.Contains(i)).ToList();
                foreach (var (dest, src) in zip(destination.axis, source.axis).OrderBy(x => x.Item1))
                {
                    perm.Insert(dest, src);
                }
            }
            else
                throw new NotImplementedException("");

            return array_ops.transpose(array, perm.ToArray());
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
        public static Tensor concat(Tensor[] values, Tensor axis, string name = "concat")
        {
            return tf.Context.ExecuteOp("ConcatV2", name, new ExecuteOpArgs(values, axis));
        }

        public static Tensor concat(object[] values, int axis, string name = "concat")
        {
            return tf.Context.ExecuteOp("ConcatV2", name, new ExecuteOpArgs(values, axis));
        }

        /// <summary>
        /// Gather slices from `params` according to `indices`. `indices` must be an integer tensor of any dimension(often 1-D).
        /// </summary>
        /// <typeparam name="T1">Element type of the indexed tensor.</typeparam>
        /// <typeparam name="T2">Element type of the index tensor.</typeparam>
        /// <param name="params">The `Tensor` from which to gather values. Must be at least rank `axis + 1`.</param>
        /// <param name="indices">The index `Tensor`.  Must be one of the following types: `int32`, `int64`. The values must be in range `[0, params.shape[axis])`.</param>
        /// <param name="name">A name for the operation (optional).</param>
        /// <param name="axis">
        /// A `Tensor`. Must be one of the following types: `int32`, `int64`. 
        /// The `axis` in `params` to gather `indices` from.Must be greater than or equal to `batch_dims`.  
        /// Defaults to the first non-batch dimension. Supports negative indexes.
        /// </param>
        /// <param name="batch_dims">An integer. The number of batch dimensions. Must be less than or equal to rank(indices).</param>
        /// <returns></returns>
        public static Tensor gather(Tensor @params, Tensor indices, string name = null, Tensor axis = null, int batch_dims = 0)
        {
            if (axis is null)
                axis = tf.convert_to_tensor(batch_dims);
            if(tensor_util.constant_value(axis) != 0)
            {
                return gen_array_ops.gather_v2(@params, indices, axis, batch_dims: batch_dims, name: name);
            }

            return gen_array_ops.gather_v2(@params, indices, axis, name: name);
        }

        public static Tensor gather(Tensor @params, Tensor indices, int axis, string name = null, int batch_dims = 0)
            => gather(@params, indices, name, ops.convert_to_tensor(axis), batch_dims);

        public static Tensor gather(ResourceVariable @params, Tensor indices, string name = null, Tensor axis = null, int batch_dims = 0)
        {
            if (axis is null)
                axis = tf.convert_to_tensor(batch_dims);
            if (tensor_util.constant_value(axis) != 0)
            {
                throw new NotImplementedException();
            }

            return @params.sparse_read(indices, name);
        }

        public static Tensor transpose<T1>(T1 a, Axis perm = null, string name = "transpose", bool conjugate = false)
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

        /// <summary>
        /// Transposes last two dimensions of tensor `a`.
        /// For example:
        /// <code> python
        ///   x = tf.constant([[1, 2, 3], [4, 5, 6]])
        ///   tf.matrix_transpose(x) # [[1, 4],
        ///                         #  [2, 5],
        ///                         #  [3, 6]]
        /// </code>
        /// Matrix with two batch dimensions.
        /// x.shape is [1, 2, 3, 4]
        /// tf.linalg.matrix_transpose(x) is shape [1, 2, 4, 3]
        /// </summary>
        /// <param name="a"></param>
        /// <param name="name"></param>
        /// <param name="conjugate"></param>
        /// <returns></returns>
        /// <exception cref="ValueError"></exception>
        public static Tensor matrix_transpose(Tensor a, string name = "matrix_transpose", bool conjugate = false)
        {
            return tf_with(ops.name_scope(name, "transpose", new { a }), scope =>
            {
                var a_shape = a.shape;
                var ndims = a.shape.ndim;
                Axis perm;
                if(ndims != 0)
                {
                    if (ndims < 2)
                    {
                        throw new ValueError("Argument `a` should be a (batch) matrix with rank " +
                            $">= 2.  Received `a` = {a} with shape: {a_shape}");
                    }
                    perm = new Axis(Enumerable.Range(0, ndims - 2).Concat(new int[] { ndims - 1, ndims - 2 }).ToArray());
                }
                else
                {
                    var a_rank = a.rank;
                    perm = new Axis(Enumerable.Range(0, a_rank - 2).Concat(new int[] { a_rank - 1, a_rank - 2 }).ToArray());
                }
                return transpose(a, perm:perm, conjugate:conjugate);
            });
        }

        public static Tensor[] split(Tensor value, int num_or_size_splits, Tensor axis = null,
            string name = "split")
        {
            return gen_array_ops.split(split_dim: axis, value: value, num_split: num_or_size_splits, name);
        }

        public static Tensor[] split(Tensor value, int[] num_or_size_splits, Tensor axis = null, int num = -1,
            string name = "split")
        {
            if(num_or_size_splits.Length == 0)
            {
                throw new ValueError("Rank-0 tensors are not supported as the num_or_size_splits argument to split.");
            }
            var size_splits = ops.convert_to_tensor(num_or_size_splits);

            if(num == -1)
            {
                num = (int)size_splits.shape[0];
            }

            return gen_array_ops.split_v(value: value, size_splits: size_splits, split_dim: axis, num_split: num, name: name);
        }

        public static Tensor slice(Tensor input, Tensor[] begin, Tensor[] size, string name = null)
               => gen_array_ops.slice(input, ops.convert_to_tensor(begin), ops.convert_to_tensor(size), name: name);

        public static Tensor slice(Tensor input, Tensor begin, Tensor size, string name = null)
            => gen_array_ops.slice(input, begin, size, name: name);


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
                var input_shape = result.op.inputs[0].shape;
                if (input_shape.ndim > -1 &&
                    !result.shape.IsFullyDefined &&
                    !(paddings_constant is null))
                {
                    var new_shape = new List<int>();
                    foreach ((NDArray padding, int dim) in zip(paddings_constant, input_shape.as_int_list()))
                    {
                        if (padding is null || dim == -1 || padding.ToArray<int>().Contains(-1))
                            new_shape.Add(-1);
                        else
                            new_shape.Add((int)np.sum(padding) + dim);
                    }
                    result.shape = new_shape.ToArray();
                }
            }

            return result;
        }

        public static Tensor placeholder(TF_DataType dtype, Shape shape = null, string name = null)
        {
            if (tf.Context.executing_eagerly())
                throw new RuntimeError("tf.placeholder() is not compatible with eager execution.");

            var _op = tf.OpDefLib._apply_op_helper("Placeholder", name: name, args: new { dtype, shape });
            return _op.output;
        }

        public static int get_positive_axis(int axis, int ndims=-100, string axis_name="axis", string ndims_name= "ndims")
        {
            if(ndims != -100)
            {
                if (axis >= 0 && axis < ndims) return axis;
                else if (-ndims <= axis && axis < 0) return axis + ndims;
                else throw new ValueError($"{axis_name}={axis} out of bounds:expected {-ndims}<={axis_name}<{ndims}");
                
            } else if(axis < 0) throw new ValueError($"{axis_name}={axis} may only be negative if {ndims_name} is statically known.");
            return axis;
        }

    }
}
