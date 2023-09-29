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

using Serilog.Debugging;
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
//using System.ComponentModel.DataAnnotations;
using System.Text;
using System.Xml.Linq;
using Tensorflow.Framework;
using Tensorflow.NumPy;
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
                return (int)_row_splits.shape[0] - 1;
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
                var row_splits = array_ops.concat(new Tensor[]
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

        public static RowPartition from_row_lengths(Tensor row_lengths,
                       bool validate=true,
                       TF_DataType dtype = TF_DataType.TF_INT32,
                       TF_DataType dtype_hint= TF_DataType.TF_INT32)
        {
            row_lengths = _convert_row_partition(
                row_lengths, "row_lengths", dtype_hint: dtype_hint, dtype: dtype);
            Tensor row_limits = math_ops.cumsum<Tensor>(row_lengths, tf.constant(-1));
            Tensor row_splits = array_ops.concat(new Tensor[] { tf.convert_to_tensor(np.array(new int[] { 0 }, TF_DataType.TF_INT64)), row_limits }, axis:0);
            return new RowPartition(row_splits: row_splits, row_lengths: row_lengths);
        }

        public static Tensor _convert_row_partition(Tensor partition, string name, TF_DataType dtype, 
            TF_DataType dtype_hint= TF_DataType.TF_INT64)
        {
            if (partition is NDArray && partition.GetDataType() == np.int32) partition = ops.convert_to_tensor(partition, name: name);
            if (partition.GetDataType() != np.int32 && partition.GetDataType() != np.int64) throw new ValueError($"{name} must have dtype int32 or int64");
            return partition;
        }

        public Tensor nrows()
        {
            /*Returns the number of rows created by this `RowPartition*/
            if (this._nrows != null) return this._nrows;
            var nsplits = tensor_shape.dimension_at_index(this._row_splits.shape, 0);
            if (nsplits == null) return array_ops.shape(this._row_splits, out_type: this.row_splits.dtype)[0] - 1;
            else return constant_op.constant(nsplits.value - 1, dtype: this.row_splits.dtype);
        }

        public Tensor row_lengths()
        {
  
            if (this._row_splits != null)
            {
                int nrows_plus_one = tensor_shape.dimension_value(this._row_splits.shape[0]);
                return tf.constant(nrows_plus_one - 1);
                
            }
            if (this._row_lengths != null)
            {
                var nrows = tensor_shape.dimension_value(this._row_lengths.shape[0]);
                return tf.constant(nrows);
            }
            if(this._nrows != null)
            {
                return tensor_util.constant_value(this._nrows);
            }
            return tf.constant(-1);
        }
    }
}
