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

using static Tensorflow.Binding;

namespace Tensorflow
{
    public class clip_ops
    {
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
    }
}
