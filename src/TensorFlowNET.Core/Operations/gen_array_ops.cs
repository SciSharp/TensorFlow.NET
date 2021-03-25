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
    public static class gen_array_ops
    {
        public static Tensor batch_to_space_nd<T>(T input, int[] block_shape, int[,] crops, string name = null)
        {
            var _op = tf.OpDefLib._apply_op_helper("BatchToSpaceND", name: name, args: new { input, block_shape, crops });

            return _op.output;
        }

        public static Tensor check_numerics(Tensor tensor, string message, string name = null)
        {
            var _op = tf.OpDefLib._apply_op_helper("CheckNumerics", name: name, args: new { tensor, message });

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
            => tf.Context.ExecuteOp("ConcatV2", name, new ExecuteOpArgs(values, axis));

        public static Tensor concat_v2(Tensor[] values, Tensor axis, string name = null)
        {
            if (tf.Context.executing_eagerly())
            {
                return concat_v2_eager_fallback(values, axis, name, tf.Context);
            }

            var _op = tf.OpDefLib._apply_op_helper("ConcatV2", name: name, args: new { values, axis });
            return _op.output;
        }

        public static Tensor concat_v2(Tensor[] values, int axis, string name = null)
            => tf.Context.ExecuteOp("ConcatV2", name, new ExecuteOpArgs(values, axis));

        private static Tensor concat_v2_eager_fallback<T1, T2>(T1[] values, T2 axis, string name, Context ctx)
        {
            var _attr_N = len(values);
            var (_attr_T, input) = tf.Runner.ArgsToMatchingEager(ctx, args: values.Select(x => (object)x).ToArray());
            var (_attr_Tidx, axis1) = tf.Runner.ArgsToMatchingEager(ctx, default_dtype: tf.int32, args: new object[] { axis });
            var _inputs_flat = input.concat(axis1);
            var _attrs = new object[] { "N", _attr_N, "T", _attr_T, "Tidx", _attr_Tidx };

            return tf.Runner.Execute(ctx, "ConcatV2", 1, _inputs_flat, _attrs, name: name)[0];
        }

        public static Tensor[] concat_offset(Tensor concat_dim, Tensor[] shape, string name = null)
        {
            var _op = tf.OpDefLib._apply_op_helper("ConcatOffset", name: name, args: new { concat_dim, shape });

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
            => tf.Context.ExecuteOp("Diag", name, new ExecuteOpArgs(diagonal));

        public static Tensor expand_dims(Tensor input, int axis, string name = null)
            => tf.Context.ExecuteOp("ExpandDims", name, new ExecuteOpArgs(input, axis)
                .SetAttributes(new { dim = axis }));

        public static Tensor gather_v2<T1, T2>(T1 @params, T2 indices, int axis, int batch_dims = 0, string name = null)
        {
            var result = tf.Context.ExecuteOp("GatherV2", name, new ExecuteOpArgs(
                @params,
                indices,
                axis).SetAttributes(new { batch_dims }));
            return result [0];
        }

        private static Tensor gather_v2_eager_fallback(object @params, object indices, int axis, string name, Context ctx)
        {
            var (_attr_T, param) = tf.Runner.ArgsToMatchingEager(ctx, args: new[] { @params });
            var (_attr_Tindice, indice) = tf.Runner.ArgsToMatchingEager(ctx, default_dtype: tf.int32, args: new[] { indices });
            var (_attr_Taxis, axiss) = tf.Runner.ArgsToMatchingEager(ctx, default_dtype: tf.int32, args: new object[] { axis });
            var _inputs_flat = param.concat(indice).concat(axiss);
            var _attrs = new object[] { "batch_dims", 0, "Tparams", _attr_T, "Tindices", _attr_Tindice, "Taxis", _attr_Taxis };

            var results = tf.Runner.Execute(ctx, "GatherV2", 1, _inputs_flat, _attrs, name: name);
            if (tf.Runner.MustRecordGradient())
                tf.Runner.RecordGradient("GatherV2", _inputs_flat, _attrs, results);
            return results[0];
        }


        public static Tensor pad(Tensor input, Tensor paddings, string name = null)
        {
            if (tf.Context.executing_eagerly())
            {
                /*var results = tf.Runner.TFE_FastPathExecute(tf.Context, tf.Context.DeviceName,
                    "Pad", name,
                    null,
                    input, paddings);
                return results[0];*/
                return pad_eager_fallback(input, paddings, name: name, ctx: tf.Context);
            }

            var _op = tf.OpDefLib._apply_op_helper("Pad", name: name, args: new { input, paddings });

            return _op.output;
        }

        private static Tensor pad_eager_fallback(Tensor inputs, Tensor padding, string name = null, Context ctx = null)
        {
            var (_attr_T, input) = tf.Runner.ArgsToMatchingEager(ctx, args: new[] { inputs });
            var (_attr_Tpaddings, paddings) = tf.Runner.ArgsToMatchingEager(ctx, default_dtype: tf.int32, args: new[] { padding });
            var _inputs_flat = input.concat(paddings);
            var _attrs = new object[] { "T", _attr_T, "Tpaddings", _attr_Tpaddings };

            var results = tf.Runner.Execute(ctx, "Pad", 1, _inputs_flat, _attrs, name: name);
            if (tf.Runner.MustRecordGradient())
                tf.Runner.RecordGradient("Pad", _inputs_flat, _attrs, results);
            return results[0];
        }

        public static Tensor pack(Tensor[] values, int axis = 0, string name = null)
            => tf.Context.ExecuteOp("Pack", name, new ExecuteOpArgs()
            {
                OpInputArgs = new object[] { values }
            }.SetAttributes(new { axis }));

        /// <summary>
        /// Return a tensor with the same shape and contents as the input tensor or value.
        /// </summary>
        /// <param name="input"></param>
        /// <param name="name"></param>
        public static Tensor identity(Tensor input, string name = null)
            => tf.Context.ExecuteOp("Identity", name, new ExecuteOpArgs(input));

        public static Tensor invert_permutation(Tensor x, string name = null)
        {
            var _op = tf.OpDefLib._apply_op_helper("InvertPermutation", name, new { x });

            return _op.outputs[0];
        }

        public static Tensor log(Tensor x, string name = null)
            => tf.Context.ExecuteOp("Log", name, new ExecuteOpArgs(x));


        public static Tensor rank(Tensor input, string name = null)
            => tf.Context.ExecuteOp("Rank", name, new ExecuteOpArgs(input));

        /// <summary>
        /// Creates a tensor filled with a scalar value.
        /// </summary>
        /// <param name="dims">A `Tensor`.</param>
        /// <param name="value">A `Tensor`. 0-D (scalar). Value to fill the returned tensor.</param>
        /// <param name="name">A name for the operation (optional).</param>
        /// <returns>A `Tensor`. Has the same type as `value`.</returns>
        public static Tensor fill<T>(Tensor dims, T value, string name = null)
            => tf.Context.ExecuteOp("Fill", name, new ExecuteOpArgs(dims, value));

        /// <summary>
        /// Return the reduction indices for computing gradients of s0 op s1 with broadcast.
        /// </summary>
        /// <param name="s0">A `Tensor`. Must be one of the following types: `int32`, `int64`.</param>
        /// <param name="s1">A `Tensor`. Must have the same type as `s0`.</param>
        /// <param name="name">A name for the operation (optional).</param>
        /// <returns>A tuple of `Tensor` objects (r0, r1).</returns>
        public static (Tensor, Tensor) broadcast_gradient_args(Tensor s0, Tensor s1, string name = "")
        {
            var results = tf.Context.ExecuteOp("BroadcastGradientArgs", name, new ExecuteOpArgs(s0, s1));
            return (results[0], results[1]);
        }

        public static Tensor reverse<T>(Tensor tensor, T axis, string name = null)
        {
            var _op = tf.OpDefLib._apply_op_helper("ReverseV2", name, new { tensor, axis });
            return _op.output;
        }

        public static Tensor reshape<T>(Tensor tensor, T shape, string name = null)
            => tf.Context.ExecuteOp("Reshape", name, new ExecuteOpArgs(tensor, shape));

        public static Tensor reshape(Tensor tensor, object[] shape, string name = null)
            => tf.Context.ExecuteOp("Reshape", name, new ExecuteOpArgs(tensor, shape));

        private static Tensor reshape_eager_fallback(Tensor tensor, object[] shape, string name, Context ctx)
        {
            var (_attr_T, _input) = tf.Runner.ArgsToMatchingEager(ctx, args: new[] { tensor });
            var (_attr_Tshape, _input_shape) = tf.Runner.ArgsToMatchingEager(ctx, args: new object[] { shape }, default_dtype: TF_DataType.TF_INT32);
            var _inputs_flat = new[] { _input[0], _input_shape[0] };
            var _attrs = new object[] { "T", _attr_T, "Tshape", _attr_Tshape };

            var results = tf.Runner.Execute(ctx, "Reshape", 1, _inputs_flat, _attrs, name: name);
            if (tf.Runner.MustRecordGradient())
            {
                tf.Runner.RecordGradient("Reshape", _inputs_flat, _attrs, results);
            }
            return results[0];
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
            var _op = tf.OpDefLib._apply_op_helper("Unique", name, new { x, out_idx });
            // TODO
            //var _result = _UniqueOutput._make(_op.outputs);
            return (_op.outputs[0], _op.outputs[1]);
        }

        public static Tensor[] unpack(Tensor value, int num, int axis = 0, string name = null)
        {
            var _op = tf.OpDefLib._apply_op_helper("Unpack", name, new { value, num, axis });
            return _op.outputs;
        }

        public static Tensor where(Tensor condition, string name = null)
        {
            var _op = tf.OpDefLib._apply_op_helper("Where", name, new { input = condition });
            return _op.output;
        }

        public static Tensor one_hot(Tensor indices, Tensor depth,
            Tensor on_value = null,
            Tensor off_value = null,
            TF_DataType dtype = TF_DataType.DtInvalid,
            int axis = -1,
            string name = null)
                => tf.Context.ExecuteOp("OneHot", name, new ExecuteOpArgs(indices, depth, on_value, off_value)
                    .SetAttributes(new { axis }));

        /// <summary>
        /// A placeholder op that passes through `input` when its output is not fed.
        /// </summary>
        /// <param name="input">The default value to produce when output is not fed.</param>
        /// <param name="shape"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public static Tensor placeholder_with_default<T>(T input, int[] shape, string name = null)
        {
            var _op = tf.OpDefLib._apply_op_helper("PlaceholderWithDefault", name, new { input, shape, name });
            return _op.outputs[0];
        }

        public static Tensor select<Tx, Ty>(Tensor condition, Tx x, Ty y, string name = null)
            => tf.Context.ExecuteOp("Select", name, new ExecuteOpArgs(condition, x, y));

        public static Tensor select_v2<Tx, Ty>(Tensor condition, Tx x, Ty y, string name = null)
            => tf.Context.ExecuteOp("SelectV2", name, new ExecuteOpArgs(condition, x, y));

        public static Tensor scatter_nd(Tensor indices, Tensor updates, Tensor[] shape, string name = null)
        {
            var _op = tf.OpDefLib._apply_op_helper("ScatterNd", name, new { indices, updates, shape });
            return _op.outputs[0];
        }

        public static Tensor shape(Tensor input, TF_DataType out_type = TF_DataType.TF_INT32, string name = null)
            => tf.Context.ExecuteOp("Shape", name, new ExecuteOpArgs(input)
                .SetAttributes(new { out_type }));

        /// <summary>
        /// Returns shape of tensors.
        /// </summary>
        /// <param name="input"></param>
        /// <param name="out_type"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public static Tensor[] shape_n(Tensor[] input, TF_DataType out_type = TF_DataType.TF_INT32, string name = null)
            => tf.Context.ExecuteOp("ShapeN", name, new ExecuteOpArgs()
            {
                OpInputArgs = new object[] { input }
            }.SetAttributes(new { out_type }));

        public static Tensor size(Tensor input, TF_DataType out_type = TF_DataType.TF_INT32, string name = null)
        {
            var _op = tf.OpDefLib._apply_op_helper("Size", name, new { input, out_type });
            return _op.outputs[0];
        }

        public static Tensor slice(Tensor input, Tensor[] begin, Tensor[] size, string name = null)
        {
            if (tf.executing_eagerly())
            {
                var result = slice_eager_fallback(input, begin, size, name, tf.Context);
                return result;
            }

            var _op = tf.OpDefLib._apply_op_helper("Slice", name, new { input, begin, size });
            return _op.outputs[0];
        }

        private static Tensor slice_eager_fallback(Tensor inputs, Tensor[] begin, Tensor[] size, string name, Context ctx)
        {
            var (_attr_T, input) = tf.Runner.ArgsToMatchingEager(ctx, args: new[] { inputs });
            var (_attr_Tidx, _inputs_Index) = tf.Runner.ArgsToMatchingEager(ctx, args: new object[] { begin, size });
            var _inputs_flat = input.concat(_inputs_Index);
            var _attrs = new object[] { "T", _attr_T, "Index", _attr_Tidx };

            var results = tf.Runner.Execute(ctx, "Slice", 1, _inputs_flat, _attrs, name: name);
            if (tf.Runner.MustRecordGradient())
            {
                tf.Runner.RecordGradient("Slice", _inputs_flat, _attrs, results);
            }
            return results[0];
        }

        public static Tensor slice<Tb, Ts>(Tensor input, Tb begin, Ts size, string name = null)
        {
            if (tf.executing_eagerly())
            {
                var outputs = tf.Runner.TFE_FastPathExecute(new FastPathOpExecInfo("Slice", name, input, begin, size));
                return outputs[0];
            }

            var _op = tf.OpDefLib._apply_op_helper("Slice", name, new { input, begin, size });
            return _op.outputs[0];
        }

        public static Tensor[] split_v(Tensor value, Tensor size_splits, 
            int axis, int num_split, string name = null)
                => tf.Context.ExecuteOp("SplitV", name, new ExecuteOpArgs(value, size_splits, axis)
                    .SetAttributes(new { num_split }));

        public static Tensor tile(Tensor input, Tensor multiples, string name = null)
            => tf.Context.ExecuteOp("Tile", name, new ExecuteOpArgs(input, multiples));

        public static Tensor tile(Tensor input, object[] multiples, string name = null)
            => tf.Context.ExecuteOp("Tile", name, new ExecuteOpArgs(input, multiples));

        public static Tensor transpose<T1>(Tensor x, T1 perm, string name = null)
            => tf.Context.ExecuteOp("Transpose", name, new ExecuteOpArgs(x, perm));

        public static Tensor ones_like(Tensor x, string name = null)
            => tf.Context.ExecuteOp("OnesLike", name, new ExecuteOpArgs(x));

        public static Tensor zeros_like(Tensor x, string name = null)
            => tf.Context.ExecuteOp("ZerosLike", name, new ExecuteOpArgs(x));

        public static Tensor stop_gradient(Tensor x, string name = null)
        {
            var _op = tf.OpDefLib._apply_op_helper("StopGradient", name, args: new { input = x, name });

            return _op.output;
        }

        public static Tensor strided_slice(Tensor input, Tensor begin, Tensor end, Tensor strides,
            long begin_mask = 0,
            long end_mask = 0,
            long ellipsis_mask = 0,
            long new_axis_mask = 0,
            long shrink_axis_mask = 0,
            string name = null)
                => tf.Context.ExecuteOp("StridedSlice", name, new ExecuteOpArgs(input, begin, end, strides)
                    .SetAttributes(new
                    {
                        begin_mask,
                        end_mask,
                        ellipsis_mask,
                        new_axis_mask,
                        shrink_axis_mask
                    }));

        public static Tensor resource_strided_slice_assign(Tensor input, Tensor begin, Tensor end, Tensor strides, Tensor value,
            int begin_mask = 0,
            int end_mask = 0,
            int ellipsis_mask = 0,
            int new_axis_mask = 0,
            int shrink_axis_mask = 0,
            string name = null)
                => tf.Context.ExecuteOp("ResourceStridedSliceAssign", name, new ExecuteOpArgs(input, begin, end, strides, value)
                    .SetAttributes(new
                    {
                        begin_mask,
                        end_mask,
                        ellipsis_mask,
                        new_axis_mask,
                        shrink_axis_mask
                    }));

        public static Tensor strided_slice<T>(Tensor input, T[] begin, T[] end, T[] strides,
            int begin_mask = 0,
            int end_mask = 0,
            int ellipsis_mask = 0,
            int new_axis_mask = 0,
            int shrink_axis_mask = 0,
            string name = null)
        {
            var _op = tf.OpDefLib._apply_op_helper("StridedSlice", name, new
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
            => tf.Context.ExecuteOp("Squeeze", name, new ExecuteOpArgs(input)
                .SetAttributes(new { squeeze_dims = axis }));

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
            var _op = tf.OpDefLib._apply_op_helper("BroadcastArgs", name, args: new { s0, s1, name });

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
            => tf.Context.ExecuteOp("BroadcastTo", name, new ExecuteOpArgs(input, shape));
    }
}
