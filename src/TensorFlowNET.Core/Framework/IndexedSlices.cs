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

namespace Tensorflow.Framework
{
    /// <summary>
    /// A sparse representation of a set of tensor slices at given indices.
    /// </summary>
    public class IndexedSlices : CompositeTensor
    {
        Tensor _values;
        public Tensor values => _values;
        Tensor _indices;
        public Tensor indices => _indices;
        Tensor _dense_shape;
        public Tensor dense_shape => _dense_shape;

        public string name => _values.name;

        public string device => _values.Device;

        public Operation op => _values.op;

        public TF_DataType dtype => _values.dtype;

        public Graph graph => _values.graph;

        public IndexedSlices(Tensor values, Tensor indices, Tensor dense_shape = null)
        {
            _values = values;
            _indices = indices;
            _dense_shape = dense_shape;

            _values.Tag = this;
        }

        public static implicit operator Tensor(IndexedSlices indexedSlices)
        {
            return _indexed_slices_to_tensor(indexedSlices);
        }

        public static implicit operator IndexedSlices(Tensor tensor)
        {
            return tensor.Tag as IndexedSlices;
        }

        /// <summary>
        /// Converts an IndexedSlices object `value` to a Tensor.
        /// </summary>
        /// <param name="indexedSlices"></param>
        /// <param name="dtype"></param>
        /// <param name="name"></param>
        /// <param name="as_ref"></param>
        /// <returns></returns>
        public static Tensor _indexed_slices_to_tensor(IndexedSlices indexedSlices, TF_DataType dtype = TF_DataType.DtInvalid, String name = "", bool as_ref = false)
        {
            return gen_math_ops.unsorted_segment_sum(indexedSlices.values, indexedSlices.indices, indexedSlices.dense_shape.slice(0));
        }
    }
}
