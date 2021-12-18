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

using System.Linq;
using static Tensorflow.Binding;

namespace Tensorflow
{
    public class clip_ops
    {
        public static (Tensors, Tensor) clip_by_global_norm(Tensor[] t_list, float clip_norm, Tensor use_norm = null, string name = null)
        {
            use_norm = global_norm(t_list, name);
            return tf_with(ops.name_scope(name, "clip_by_global_norm", t_list), delegate
            {
                // Calculate L2-norm, clip elements by ratio of clip_norm to L2-norm
                var scale_for_finite = clip_norm * math_ops.minimum(
                    1.0f / use_norm,
                    constant_op.constant(1.0, dtype: use_norm.dtype) / clip_norm);

                // If use_norm is any finite number, this is a no-op. For inf/-inf/NaN,
                // this will make scale NaN.
                var scale = scale_for_finite + (use_norm - use_norm);

                Tensors values_clipped = new Tensors();
                foreach (var (i, v) in enumerate(t_list))
                    values_clipped.Add(array_ops.identity(v * scale, name: $"{name}_{i}"));
                return (values_clipped, use_norm);
            });
        }

        public static Tensor clip_by_value<T1, T2>(Tensor t, T1 clip_value_min, T2 clip_value_max, string name = null)
        {
            return tf_with(ops.name_scope(name, "clip_by_value", new { t, clip_value_min, clip_value_max }), delegate
            {
                var values = ops.convert_to_tensor(t, name: "t");
                // Go through list of tensors, for each value in each tensor clip
                var t_min = math_ops.minimum(values, clip_value_max);
                // Assert that the shape is compatible with the initial shape,
                // to prevent unintentional broadcasting.
                _ = values.shape.merge_with(t_min.shape);
                var t_max = math_ops.maximum(t_min, clip_value_min, name: name);
                _ = values.shape.merge_with(t_max.shape);

                return t_max;
            });
        }

        /// <summary>
        /// Computes the global norm of multiple tensors.
        /// </summary>
        /// <param name="t_list"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public static Tensor global_norm(Tensor[] t_list, string name = null)
        {
            return tf_with(ops.name_scope(name, "global_norm", t_list), delegate
            {
                var half_squared_norms = t_list.Select(v => nn_ops.l2_loss(v)).ToArray();
                var half_squared_norm = math_ops.reduce_sum(array_ops.stack(half_squared_norms));
                var norm = math_ops.sqrt(half_squared_norm * 
                    constant_op.constant(2.0, dtype: half_squared_norm.dtype),
                    name: "global_norm");
                return norm;
            });
        }
    }
}
