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
using Tensorflow.Framework;
using static Tensorflow.Binding;

namespace Tensorflow
{
    /// <summary>
    /// Partitioning of a sequence of values into contiguous subsequences ("rows").
    /// </summary>
    public class RowPartition : CompositeTensor
    {
        Tensor _row_splits;
        public Tensor row_splits => _row_splits;
        Tensor _row_lengths;
        Tensor _value_rowids;
        Tensor _nrows;

        public int static_nrows
        {
            get
            {
                return _row_splits.shape[0] - 1;
            }
        }

        public int static_uniform_row_length
        {
            get
            {
                return -1;
            }
        }

        public RowPartition(Tensor row_splits, 
            Tensor row_lengths = null, Tensor value_rowids = null, Tensor nrows = null,
            Tensor uniform_row_length = null)
        {
            _row_splits = row_splits;
            _row_lengths = row_lengths;
            _value_rowids = value_rowids;
            _nrows = nrows;
        }

        /// <summary>
        /// Creates a `RowPartition` with rows partitioned by `value_rowids`.
        /// </summary>
        /// <param name="value_rowids"></param>
        /// <param name="nrows"></param>
        /// <param name="validate"></param>
        /// <param name="preferred_dtype"></param>
        /// <returns></returns>
        public static RowPartition from_value_rowids(Tensor value_rowids,
            Tensor nrows = null, bool validate = true, TF_DataType preferred_dtype = TF_DataType.DtInvalid)
        {
            return tf_with(ops.name_scope(null, "RowPartitionFromValueRowIds"), scope =>
            {
                var value_rowids_int32 = math_ops.cast(value_rowids, dtypes.int32);
                var nrows_int32 = math_ops.cast(nrows, dtypes.int32);
                var row_lengths = tf.math.bincount(value_rowids_int32, 
                    minlength: nrows_int32,
                    maxlength: nrows_int32,
                    dtype: value_rowids.dtype);
                var row_splits = array_ops.concat(new object[]
                {
                    ops.convert_to_tensor(new long[] { 0 }),
                    tf.cumsum(row_lengths)
                }, axis: 0);

                return new RowPartition(row_splits,
                    row_lengths: row_lengths,
                    value_rowids: value_rowids,
                    nrows: nrows);
            });
        }

        public static RowPartition from_row_splits(Tensor row_splits,
            bool validate = true, TF_DataType preferred_dtype = TF_DataType.DtInvalid)
        {
            return tf_with(ops.name_scope(null, "RowPartitionFromRowSplits"), scope =>
            {
                return new RowPartition(row_splits);
            });
        }
    }
}
