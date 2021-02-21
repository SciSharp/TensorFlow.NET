/*****************************************************************************
   Copyright 2021 Haiping Chen. All Rights Reserved.

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
using System.Text;
using System.Linq;
using Tensorflow.Framework;
using static Tensorflow.Binding;

namespace Tensorflow
{
    /// <summary>
    /// Represents a ragged tensor.
    /// </summary>
    public class RaggedTensor : CompositeTensor
    {
        Tensor _values;
        RowPartition _row_partition;
        public TF_DataType dtype => _values.dtype;
        public TensorShape shape
        {
            get
            {
                var nrows = _row_partition.static_nrows;
                var ncols = _row_partition.static_uniform_row_length;
                return new TensorShape(nrows, ncols);
            }
        }

        public RaggedTensor(Tensor values,
            bool @internal = true,
            RowPartition row_partition = null)
        {
            _values = values;
            _row_partition = row_partition;
        }

        public static RaggedTensor from_row_partition(Tensor values, RowPartition row_partition, bool validate = true)
        {
            return new RaggedTensor(values, @internal: true, row_partition: row_partition);
        }

        /// <summary>
        /// Creates a `RaggedTensor` with rows partitioned by `value_rowids`.
        /// </summary>
        /// <param name="values"></param>
        /// <param name="value_rowids"></param>
        /// <param name="nrows"></param>
        /// <param name="name"></param>
        /// <param name="validate"></param>
        /// <returns></returns>
        public static RaggedTensor from_value_rowids(Tensor values, Tensor value_rowids, 
            Tensor nrows = null, string name = null, bool validate = true)
        {
            return tf_with(ops.name_scope(name, "RaggedFromValueRowIds"), scope =>
            {
                var row_partition = RowPartition.from_value_rowids(value_rowids,
                  nrows: nrows,
                  validate: validate);
                return from_row_partition(values, row_partition, validate: validate);
            });
        }

        public override string ToString()
            => $"tf.RaggedTensor: shape={shape} [{string.Join(", ", _values.StringData().Take(10))}]";

        public static implicit operator Tensor(RaggedTensor indexedSlices)
        {
            return indexedSlices._values;
        }

        public static implicit operator RaggedTensor(Tensor tensor)
        {
            return tensor.Tag as RaggedTensor;
        }
    }
}
