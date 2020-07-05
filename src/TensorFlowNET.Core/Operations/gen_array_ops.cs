﻿/*****************************************************************************
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
using System.Collections.Generic;
using static Tensorflow.Binding;
using Tensorflow.Eager;
using System.Linq;
#pragma warning disable CS0105 // Using directive appeared previously in this namespace
using static Tensorflow.Binding;
#pragma warning restore CS0105 // Using directive appeared previously in this namespace

namespace Tensorflow
{
    public static class gen_array_ops
    {
        public static Tensor batch_to_space_nd<T>(T input, int[] block_shape, int[,] crops, string name = null)
        {
            var _op = tf._op_def_lib._apply_op_helper("BatchToSpaceND", name: name, args: new { input, block_shape, crops });

            return _op.output;
        }
        
        public static Tensor check_numerics(Tensor tensor, string message, string name = null)
        {
            var _op = tf._op_def_lib._apply_op_helper("CheckNumerics", name: name, args: new { tensor, message });

            return _op.output;
        }

        /// <summary>
        /// Concatenates tensors along one dimension.
        /// </summary>
        /// <param name="values"></param>
        /// <param name="axis"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public static Tensor concat_v2<T, Ta>(T[] values, Ta axis, string name = null)
        {
            if (tf.context.executing_eagerly())
            {
                var results = tf.Runner.TFE_FastPathExecute(tf.context, tf.context.device_name,
                    "ConcatV2", name, 
                    null,
                    values, axis);

                return results[0];
            }

            var _op = tf._op_def_lib._apply_op_helper("ConcatV2", name: name, args: new { values, axis });
            return _op.output;
        }

        public static Tensor concat_v2(Tensor[] values, Tensor axis, string name = null)
        {
            if (tf.context.executing_eagerly())
            {
                return concat_v2_eager_fallback(values, axis, name, tf.context);
            }

            var _op = tf._op_def_lib._apply_op_helper("ConcatV2", name: name, args: new { values, axis });
            return _op.output;
        }

        private static Tensor concat_v2_eager_fallback<T1, T2>(T1[] values, T2 axis, string name, Context ctx)
        {
            var _attr_N = len(values);
            var (_attr_T, input) = tf._execute.args_to_matching_eager(ctx, args: values.Select(x => (object)x).ToArray());
            var (_attr_Tidx, axis1) = tf._execute.args_to_matching_eager(ctx, default_dtype: tf.int32, args: new object[] { axis });
            var _inputs_flat = input.concat(axis1);
            var _attrs = new object[] { "N", _attr_N, "T", _attr_T, "Tidx", _attr_Tidx };

            return tf._execute.execute(ctx, "ConcatV2", 1, _inputs_flat, _attrs, name: name)[0];
        }

        public static Tensor[] concat_offset(Tensor concat_dim, Tensor[] shape, string name = null)
        {
            var _op = tf._op_def_lib._apply_op_helper("ConcatOffset", name: name, args: new { concat_dim, shape });

            return _op.outputs;
        }

        /// <summary>
        ///    Returns a diagonal tensor with a given diagonal values.
        /// </summary>
        /// <param name="diagonal">
        ///    Rank k tensor where k is at most 1.
        /// </param>
        /// <param name="name">
        /// If specified, the created operation in the graph will be this one, otherwise it will be named 'Diag'.
        /// </param>
        /// <returns>
        ///    The Operation can be fetched from the resulting Tensor, by fetching the Operation property from the result.
        /// </returns>
        /// <remarks>
        ///    Given a <c>diagonal</c>, this operation returns a tensor with the <c>diagonal</c> and
        ///    everything else padded with zeros. The diagonal is computed as follows:
        ///    
        ///    Assume <c>diagonal</c> has dimensions [D1,..., Dk], then the output is a tensor of
        ///    rank 2k with dimensions [D1,..., Dk, D1,..., Dk] where:
        ///    
        ///    <c>output[i1,..., ik, i1,..., ik] = diagonal[i1, ..., ik]</c> and 0 everywhere else.
        ///    
        ///    For example:
        ///    
        ///   <code>
        ///    # 'diagonal' is [1, 2, 3, 4]
        ///    tf.diag(diagonal) ==&amp;gt; [[1, 0, 0, 0]
        ///    [0, 2, 0, 0]
        ///    [0, 0, 3, 0]
        ///    [0, 0, 0, 4]]
        ///   </code>
        /// </remarks>
        public static Tensor diag(Tensor diagonal, string name = null)
        {
            if (tf.context.executing_eagerly())
            {
                var results = tf.Runner.TFE_FastPathExecute(tf.context, tf.context.device_name,
                    "Diag", name,
                    null,
                    diagonal);

                return results[0];
            }

            var op = tf._op_def_lib._apply_op_helper("Diag", name: name, args: new { diagonal });

            return op.output;
        }

        public static Tensor expand_dims(Tensor input, int axis, string name = null)
        {
            if (tf.context.executing_eagerly())
            {
                var results = tf.Runner.TFE_FastPathExecute(tf.context, tf.context.device_name,
                    "ExpandDims", name,
                    null,
                    input, tf.convert_to_tensor(axis));

                return results[0];
            }

            var _op = tf._op_def_lib._apply_op_helper("ExpandDims", name: name, args: new { input, dim = axis });

            return _op.outputs[0];
        }

        public static Tensor gather_v2<T1, T2>(T1 @params, T2 indices, int axis, string name = null)
        {
            var _op = tf._op_def_lib._apply_op_helper("GatherV2", name: name, new { @params, indices, axis });

            return _op.outputs[0];
        }

        public static Tensor pad(Tensor input, Tensor paddings, string name = null)
        {
            var _op = tf._op_def_lib._apply_op_helper("Pad", name: name, args: new { input, paddings });

            return _op.output;
        }

        public static Tensor pack(Tensor[] values, int axis = 0, string name = null)
        {
            if(tf.context.executing_eagerly())
            {
                var results = tf.Runner.TFE_FastPathExecute(tf.context, tf.context.device_name, 
                    "Pack", name,
                    null,
                    values,
                    "axis", axis);
                return results[0];
            }

            var _op = tf._op_def_lib._apply_op_helper("Pack", name: name, args: new { values, axis });
            return _op.output;
        }

        public static Tensor placeholder(TF_DataType dtype, TensorShape shape = null, string name = null)
        {
            var _op = tf._op_def_lib._apply_op_helper("Placeholder", name: name, args: new { dtype, shape });
            var _result = _op.outputs;
            var _inputs_flat = _op.inputs;

            var _attrs = new Dictionary<string, object>();
            _attrs["dtype"] = _op.get_attr("dtype");
            _attrs["shape"] = _op.get_attr("shape");

            return new Tensor(_op, 0, dtype);
        }

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
            var op = tf._op_def_lib._apply_op_helper("PreventGradient", name: name, args: new { input, message });
            return op.output;
        }

        /// <summary>
        /// Return a tensor with the same shape and contents as the input tensor or value.
        /// </summary>
        /// <param name="input"></param>
        /// <param name="name"></param>
        public static Tensor identity(Tensor input, string name = null)
        {
            if (tf.context.executing_eagerly())
            {
                var results = tf.Runner.TFE_FastPathExecute(tf.context, tf.context.device_name,
                    "Identity", name,
                    null,
                    input);

                return results[0];
            }

            var _op = tf._op_def_lib._apply_op_helper("Identity", name, new { input });

            return _op.output;
        }

        public static Tensor invert_permutation(Tensor x, string name = null)
        {
            var _op = tf._op_def_lib._apply_op_helper("InvertPermutation", name, new { x });

            return _op.outputs[0];
        }

        public static Tensor log(Tensor x, string name = null)
        {
            var _op = tf._op_def_lib._apply_op_helper("Log", name: name, args: new { x });

            return _op.outputs[0];
        }

        public static Tensor rank(Tensor input, string name = null)
        {
            var _op = tf._op_def_lib._apply_op_helper("Rank", name: name, args: new { input });

            return _op.outputs[0];
        }

        /// <summary>
        /// Creates a tensor filled with a scalar value.
        /// </summary>
        /// <param name="dims">A `Tensor`.</param>
        /// <param name="value">A `Tensor`. 0-D (scalar). Value to fill the returned tensor.</param>
        /// <param name="name">A name for the operation (optional).</param>
        /// <returns>A `Tensor`. Has the same type as `value`.</returns>
        public static Tensor fill<T>(Tensor dims, T value, string name = null)
        {
            if (tf.context.executing_eagerly())
            {
                var results = tf.Runner.TFE_FastPathExecute(tf.context, tf.context.device_name,
                    "Fill", name,
                    null,
                    dims, value);

                return results[0];
            }

            var _op = tf._op_def_lib._apply_op_helper("Fill", name, new { dims, value });
            return _op.output;
        }

        /// <summary>
        /// Return the reduction indices for computing gradients of s0 op s1 with broadcast.
        /// </summary>
        /// <param name="s0">A `Tensor`. Must be one of the following types: `int32`, `int64`.</param>
        /// <param name="s1">A `Tensor`. Must have the same type as `s0`.</param>
        /// <param name="name">A name for the operation (optional).</param>
        /// <returns>A tuple of `Tensor` objects (r0, r1).</returns>
        public static (Tensor, Tensor) broadcast_gradient_args(Tensor s0, Tensor s1, string name = "")
        {
            if (tf.context.executing_eagerly())
            {
                var results = tf.Runner.TFE_FastPathExecute(tf.context, tf.context.device_name,
                    "BroadcastGradientArgs", name, 
                    null,
                    s0,s1);
                
                return (results[0], results[1]);
            }

            var _op = tf._op_def_lib._apply_op_helper("BroadcastGradientArgs", name, new { s0, s1 });

            return (_op.outputs[0], _op.outputs[1]);
        }

        public static Tensor reverse<T>(Tensor tensor, T axis, string name = null)
        {
            var _op = tf._op_def_lib._apply_op_helper("ReverseV2", name, new { tensor, axis });
            return _op.output;
        }

        public static Tensor reshape<T1, T2>(T1 tensor, T2 shape, string name = null)
        {
            if (tf.context.executing_eagerly())
            {
                var results = tf.Runner.TFE_FastPathExecute(tf.context, tf.context.device_name,
                    "Reshape", name, 
                    null,
                    tensor, shape);

                return results[0];
            }

            var _op = tf._op_def_lib._apply_op_helper("Reshape", name, new { tensor, shape });
            return _op.output;
        }

        public static Tensor reshape(Tensor tensor, int[] shape, string name = null)
        {
            var _op = tf._op_def_lib._apply_op_helper("Reshape", name, new { tensor, shape });
            return _op.outputs[0];
        }

        /// <summary>
        /// Finds unique elements in a 1-D tensor.
        /// </summary>
        /// <param name="x"></param>
        /// <param name="out_idx"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public static (Tensor, Tensor) unique(Tensor x, TF_DataType out_idx = TF_DataType.TF_INT32, string name = null)
        {
            var _op = tf._op_def_lib._apply_op_helper("Unique", name, new { x, out_idx });
            // TODO
            //var _result = _UniqueOutput._make(_op.outputs);
            return (_op.outputs[0], _op.outputs[1]);
        }

        public static Tensor[] unpack(Tensor value, int num, int axis = 0, string name = null)
        {
            var _op = tf._op_def_lib._apply_op_helper("Unpack", name, new { value, num, axis });
            return _op.outputs;
        }

        public static Tensor where(Tensor condition, string name = null)
        {
            var _op = tf._op_def_lib._apply_op_helper("Where", name, new { input = condition });
            return _op.output;
        }

        public static Tensor one_hot(Tensor indices, Tensor depth,
            Tensor on_value = null,
            Tensor off_value = null,
            TF_DataType dtype = TF_DataType.DtInvalid, 
            int axis = -1,
            string name = null)
        {
            if (tf.context.executing_eagerly())
            {
                var results = tf.Runner.TFE_FastPathExecute(tf.context, tf.context.device_name,
                    "OneHot", name,
                    null,
                    indices, depth, on_value, off_value,
                    "axis", axis);

                return results[0];
            }

            var _op = tf._op_def_lib._apply_op_helper("OneHot", name, new { indices, depth, on_value, off_value, axis });
            return _op.outputs[0];
        }

        /// <summary>
        /// A placeholder op that passes through `input` when its output is not fed.
        /// </summary>
        /// <param name="input">The default value to produce when output is not fed.</param>
        /// <param name="shape"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public static Tensor placeholder_with_default<T>(T input, int[] shape, string name = null)
        {
            var _op = tf._op_def_lib._apply_op_helper("PlaceholderWithDefault", name, new { input, shape, name });
            return _op.outputs[0];
        }

        public static Tensor select<Tx, Ty>(Tensor condition, Tx t, Ty e, string name = null)
        {
            if (tf.context.executing_eagerly())
            {
                var results = tf.Runner.TFE_FastPathExecute(tf.context, tf.context.device_name,
                    "SelectV2", name,
                    null,
                    condition, t, e);

                return results[0];
            }

            var _op = tf._op_def_lib._apply_op_helper("Select", name, new { condition, t, e });
            return _op.outputs[0];
        }

        public static Tensor scatter_nd(Tensor indices, Tensor updates, Tensor[] shape, string name = null)
        {
            var _op = tf._op_def_lib._apply_op_helper("ScatterNd", name, new { indices, updates, shape });
            return _op.outputs[0];
        }

        public static Tensor shape(Tensor input, TF_DataType out_type = TF_DataType.TF_INT32, string name = null)
        {
            if (tf.context.executing_eagerly())
            {
                var results = tf.Runner.TFE_FastPathExecute(tf.context, tf.context.device_name,
                    "Shape", name, 
                    null,
                    input,
                    "out_type", out_type);

                return results[0];
            }

            var _op = tf._op_def_lib._apply_op_helper("Shape", name, new { input, out_type });
            return _op.outputs[0];
        }

        /// <summary>
        /// Returns shape of tensors.
        /// </summary>
        /// <param name="input"></param>
        /// <param name="out_type"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public static Tensor[] shape_n(Tensor[] input, TF_DataType out_type = TF_DataType.TF_INT32, string name = null)
        {
            var _op = tf._op_def_lib._apply_op_helper("ShapeN", name, new { input, out_type });
            return _op.outputs;
        }

        public static Tensor size(Tensor input, TF_DataType out_type = TF_DataType.TF_INT32, string name = null)
        {
            var _op = tf._op_def_lib._apply_op_helper("Size", name, new { input, out_type });
            return _op.outputs[0];
        }

        /// <summary>
        /// Return a slice from 'input'
        /// </summary>
        /// <param name="input"></param>
        /// <param name="begin"></param>
        /// <param name="size"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public static Tensor slice(Tensor input, Tensor begin, Tensor size, string name = null)
        {
            var _op = tf._op_def_lib._apply_op_helper("Slice", name, new { input, begin, size });
            return _op.outputs[0];
        }

        public static Tensor tile<T>(Tensor input, T multiples, string name = null)
        {
            if (tf.context.executing_eagerly())
            {
                var results = tf.Runner.TFE_FastPathExecute(tf.context, tf.context.device_name,
                    "Tile", name, 
                    null,
                    input, multiples);

                return results[0];
            }

            var _op = tf._op_def_lib._apply_op_helper("Tile", name, new { input, multiples });
            return _op.outputs[0];
        }

        public static Tensor transpose<T1, T2>(T1 x, T2 perm, string name = null)
        {
            if (tf.context.executing_eagerly())
            {
                var results = tf.Runner.TFE_FastPathExecute(tf.context, tf.context.device_name,
                    "Transpose", name,
                    null,
                    x, perm);

                return results[0];
            }
            var _op = tf._op_def_lib._apply_op_helper("Transpose", name, new { x, perm });
            return _op.outputs[0];
        }

        public static Tensor zeros_like(Tensor x, string name = null)
        {
            var _op = tf._op_def_lib._apply_op_helper("ZerosLike", name, new { x });
            return _op.outputs[0];
        }

        public static Tensor stop_gradient(Tensor x, string name = null)
        {
            var _op = tf._op_def_lib._apply_op_helper("StopGradient", name, args: new { input = x, name });

            return _op.output;
        }

        public static Tensor strided_slice(Tensor input, Tensor begin, Tensor end, Tensor strides,
            int begin_mask = 0,
            int end_mask = 0,
            int ellipsis_mask = 0,
            int new_axis_mask = 0,
            int shrink_axis_mask = 0,
            string name = null)
        {
            if (tf.context.executing_eagerly())
            {
                var results = tf.Runner.TFE_FastPathExecute(tf.context, tf.context.device_name,
                    "StridedSlice", name, 
                    null,
                    input, begin, end, strides,
                    "begin_mask", begin_mask,
                    "end_mask", end_mask,
                    "ellipsis_mask", ellipsis_mask,
                    "new_axis_mask", new_axis_mask,
                    "shrink_axis_mask", shrink_axis_mask);
                
                return results[0];
            }

            var _op = tf._op_def_lib._apply_op_helper("StridedSlice", name, new
            {
                input,
                begin,
                end,
                strides,
                begin_mask,
                end_mask,
                ellipsis_mask,
                new_axis_mask,
                shrink_axis_mask
            });

            return _op.outputs[0];
        }

        public static Tensor strided_slice<T>(Tensor input, T[] begin, T[] end, T[] strides,
            int begin_mask = 0,
            int end_mask = 0,
            int ellipsis_mask = 0,
            int new_axis_mask = 0,
            int shrink_axis_mask = 0,
            string name = null)
        {
            var _op = tf._op_def_lib._apply_op_helper("StridedSlice", name, new
            {
                input,
                begin,
                end,
                strides,
                begin_mask,
                end_mask,
                ellipsis_mask,
                new_axis_mask,
                shrink_axis_mask
            });

            return _op.outputs[0];
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
            int begin_mask = 0, int end_mask = 0, int ellipsis_mask = 0, int new_axis_mask = 0,
            int shrink_axis_mask = 0, string name = null)
        {
            var op = tf._op_def_lib._apply_op_helper("StridedSliceGrad", name: name, args: new
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
            });

            return op.output;
        }

        public static Tensor slice<Tb, Ts>(Tensor input, Tb begin, Ts size, string name = null)
        {
            var _op = tf._op_def_lib._apply_op_helper("Slice", name, new { input, begin, size });
            return _op.outputs[0];
        }

        /// <summary>
        /// Removes dimensions of size 1 from the shape of a tensor.
        /// Given a tensor `input`, this operation returns a tensor of the same type with
        /// all dimensions of size 1 removed.If you don't want to remove all size 1
        /// dimensions, you can remove specific size 1 dimensions by specifying
        /// `axis`.
        /// </summary>
        /// <param name="input"> A `Tensor`. The `input` to squeeze.</param>
        /// <param name="axis"> An optional list of `ints`. Defaults to `[]`. If specified, only squeezes the dimensions listed.</param>
        /// <param name="name"> A name for the operation (optional).</param>
        /// <returns> A `Tensor`. Has the same type as `input`.</returns>
        public static Tensor squeeze(Tensor input, int[] axis = null, string name = null)
        {
            if (tf.context.executing_eagerly())
            {
                var results = tf.Runner.TFE_FastPathExecute(tf.context, tf.context.device_name,
                    "Squeeze", name, 
                    null,
                    input,
                    "squeeze_dims", axis);

                return results[0];
            }

            if (axis == null) axis = new int[0];
            var _op = tf._op_def_lib._apply_op_helper("Squeeze", name, args: new { input, squeeze_dims = axis });

            return _op.outputs[0];
        }

        /// <summary>
        /// Return the shape of s0 op s1 with broadcast.
        /// Given `s0` and `s1`, tensors that represent shapes, compute `r0`, the
        /// broadcasted shape. `s0`, `s1` and `r0` are all integer vectors.
        /// </summary>
        /// <param name="s0"> A `Tensor`. Must be one of the following types: `int32`, `int64`.</param>
        /// <param name="s1"> A `Tensor`. Must have the same type as `s0`.</param>
        /// <param name="name"> A name for the operation (optional).</param>
        /// <returns> `Tensor`. Has the same type as `s0`.</returns>
        public static Tensor broadcast_args(Tensor s0, Tensor s1, string name = null)
        {
            var _op = tf._op_def_lib._apply_op_helper("BroadcastArgs", name, args: new { s0, s1, name });

            return _op.outputs[0];
        }

        /// <summary>
        /// Broadcast an array for a compatible shape.
        /// </summary>
        /// <param name="input"></param>
        /// <param name="shape"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public static Tensor broadcast_to<T>(Tensor input, T shape, string name = null)
        {
            if (tf.context.executing_eagerly())
            {
                var results = tf.Runner.TFE_FastPathExecute(tf.context, tf.context.device_name,
                    "BroadcastTo", name,
                    null,
                    input, shape);

                return results[0];
            }

            var _op = tf._op_def_lib._apply_op_helper("BroadcastTo", name, args: new { input, shape, name });

            return _op.outputs[0];
        }
    }
}
