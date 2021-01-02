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

namespace Tensorflow
{
    public partial class tensorflow
    {
        public Tensor convert_to_tensor(object value, TF_DataType dtype = TF_DataType.DtInvalid, string name = null, TF_DataType preferred_dtype = TF_DataType.DtInvalid)
            => ops.convert_to_tensor(value, dtype, name, preferred_dtype: preferred_dtype);

        public Tensor strided_slice(Tensor input, Tensor begin, Tensor end, Tensor strides = null,
            int begin_mask = 0,
            int end_mask = 0,
            int ellipsis_mask = 0,
            int new_axis_mask = 0,
            int shrink_axis_mask = 0,
            string name = null) => gen_array_ops.strided_slice(input: input,
                begin: begin,
                end: end,
                strides: strides,
                begin_mask: begin_mask,
                end_mask: end_mask,
                ellipsis_mask: ellipsis_mask,
                new_axis_mask: new_axis_mask,
                shrink_axis_mask: shrink_axis_mask,
                name: name);

        public Tensor strided_slice<T>(Tensor input, T[] begin, T[] end, T[] strides = null,
            int begin_mask = 0,
            int end_mask = 0,
            int ellipsis_mask = 0,
            int new_axis_mask = 0,
            int shrink_axis_mask = 0,
            string name = null) => gen_array_ops.strided_slice(input: input,
                begin: begin,
                end: end,
                strides: strides,
                begin_mask: begin_mask,
                end_mask: end_mask,
                ellipsis_mask: ellipsis_mask,
                new_axis_mask: new_axis_mask,
                shrink_axis_mask: shrink_axis_mask,
                name: name);

        /// <summary>
        /// Splits a tensor into sub tensors.
        /// </summary>
        /// <param name="value">The Tensor to split.</param>
        /// <param name="num_split">Either an integer indicating the number of splits along split_dim or a 1-D integer
        /// Tensor or Python list containing the sizes of each output tensor along split_dim.
        /// If a scalar then it must evenly divide value.shape[axis]; otherwise the sum of sizes along the split dimension must match that of the value.</param>
        /// <param name="axis">An integer or scalar int32 Tensor. The dimension along which to split. Must be in the range [-rank(value), rank(value)). Defaults to 0.</param>
        /// <param name="name">A name for the operation (optional)</param>
        /// <returns>if num_or_size_splits is a scalar returns num_or_size_splits Tensor objects;
        /// if num_or_size_splits is a 1-D Tensor returns num_or_size_splits.get_shape[0] Tensor objects resulting from splitting value.</returns>
        public Tensor[] split(Tensor value, int num_split, Tensor axis, string name = null)
            => array_ops.split(
                value: value,
                num_split: num_split,
                axis: axis,
                name: name);

        public Tensor[] split(Tensor value, int num_split, int axis, string name = null)
            => array_ops.split(
                value: value,
                num_split: num_split,
                axis: axis,
                name: name);
    }
}
